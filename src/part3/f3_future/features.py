"""As-of-T graph features, with optional F1 and survey inputs.

The 18 feature positions follow the F3 specification; x15 is deliberately
redefined as unsupervised historical venue prestige (not future target encoding).
"""
import math
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


FEATURE_COLUMNS = [f"x{i:02d}" for i in range(1, 19)]


def _power_pagerank(A, ages, tau=None, alpha=.85, steps=50):
    n = A.shape[0]
    if n == 0:
        return np.array([])
    if tau is None:
        p = np.full(n, 1.0 / n)
    else:
        scaled = -ages / tau
        scaled -= scaled.max()
        p = np.exp(scaled)
        p /= p.sum()
    outgoing = np.asarray(A.sum(axis=1)).ravel()
    inv_out = np.divide(1.0, outgoing, out=np.zeros_like(outgoing), where=outgoing > 0)
    state = p.copy()
    dangling = outgoing == 0
    for _ in range(steps):
        nxt = alpha * (A.T @ (state * inv_out))
        nxt += (1-alpha + alpha * state[dangling].sum()) * p
        state = nxt
    return state


def _hits_authority(A, steps=100):
    n = A.shape[0]
    if n == 0:
        return np.array([])
    score = np.ones(n, dtype=np.float64) / np.sqrt(n)
    for _ in range(steps):
        score = A.T @ (A @ score)
        norm = np.linalg.norm(score)
        if norm == 0:
            return np.zeros(n)
        score /= norm
    return score


class NoveltyModel:
    """Frozen train-year TF-IDF + TruncatedSVD, nearest earlier papers only."""
    def __init__(self, n_components=32, max_features=30000, enabled=True):
        self.enabled = enabled
        self.vectorizer = None
        self.svd = None
        self.n_components = n_components
        self.max_features = max_features
        self.vectors = {}

    @staticmethod
    def text(d):
        return (str(d.get("title", "")) + " " + str(d.get("abstract", ""))).strip()

    def fit(self, G_train):
        if not self.enabled:
            print("[TEXT] Novelty disabled, x17=0", flush=True)
            return self
        docs = [self.text(a) for _, a in G_train.nodes(data=True)]
        valid = [d for d in docs if d]
        if len(valid) < 3:
            print("[TEXT] Too few documents for LSA; x17=0", flush=True)
            self.enabled = False
            return self
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features,
                                          min_df=1, dtype=np.float32)
        X = self.vectorizer.fit_transform(docs)
        dim = min(self.n_components, X.shape[0]-1, X.shape[1]-1)
        if dim < 2:
            self.enabled = False
            return self
        self.svd = TruncatedSVD(n_components=dim, n_iter=5, random_state=42)
        self.svd.fit(X)
        print(f"[TEXT] Fitted TF-IDF({X.shape[1]}) + LSA({dim}) only on train graph", flush=True)
        return self

    def prepare(self, G):
        if not self.enabled:
            return
        # Transform in batches to cap temporary TF-IDF memory.
        nodes = list(G.nodes())
        for start in range(0, len(nodes), 3000):
            chunk = [node for node in nodes[start:start+3000] if node not in self.vectors]
            if not chunk:
                continue
            X = self.vectorizer.transform([self.text(G.nodes[n]) for n in chunk])
            V = normalize(self.svd.transform(X)).astype(np.float32)
            self.vectors.update(zip(chunk, V))

    def calculate(self, G, candidate_ids):
        result = {pid: 0.0 for pid in candidate_ids}
        if not self.enabled:
            return result
        # Each publication-year compares only to papers of the preceding two years.
        byyear = defaultdict(list)
        for n in G.nodes():
            byyear[G.nodes[n]["year"]].append(n)
        candidate_byyear = defaultdict(list)
        for n in candidate_ids:
            candidate_byyear[G.nodes[n]["year"]].append(n)
        for y, cand in candidate_byyear.items():
            earlier = byyear[y-2] + byyear[y-1]
            if not earlier:
                continue
            vectors_past = np.array([self.vectors[n] for n in earlier])
            vectors_cand = np.array([self.vectors[n] for n in cand])
            knn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=1)
            knn.fit(vectors_past)
            dist, _ = knn.kneighbors(vectors_cand)
            for pid, distance in zip(cand, dist[:, 0]):
                result[pid] = float(np.clip(distance, 0, 1))
        return result


def extract_features(G, T, candidate_ids, novelty=None, f1_weights=None, survey_scores=None):
    """No future labels or citation edges are read here. All features as of T."""
    nodelist = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodelist)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=np.float64, format="csr")
    if not sparse.issparse(A):
        A = sparse.csr_array(A)
    indeg = np.asarray(A.sum(axis=0)).ravel()
    outdeg = np.asarray(A.sum(axis=1)).ravel()
    age = np.array([max(0, T-G.nodes[n]["year"]) for n in nodelist], dtype=float)
    pr = _power_pagerank(A, age, tau=None)
    citerank = _power_pagerank(A, age, tau=3.0)
    authority = _hits_authority(A)
    print(f"[FEATURE T={T}] centrality ready; computing k-core...", flush=True)
    kcore = nx.core_number(G.to_undirected(as_view=True)) if len(G) else {}

    # Annual *incoming* citation counts; source publication year marks citation year.
    peryear = defaultdict(Counter)
    core_weight = Counter()
    citing_domains = defaultdict(Counter)
    venue_aggregate = defaultdict(list)
    for src, tgt in G.edges():
        peryear[tgt][G.nodes[src]["year"]] += 1
        citing_domains[tgt][G.nodes[src]["main_domain"]] += 1
        if f1_weights is not None:
            core_weight[tgt] += max(0.0, float(f1_weights.get((src, tgt), 0.0)))
    for u in nodelist:
        if G.nodes[u]["year"] <= T - 3:
            venue = G.nodes[u]["venue"].strip().lower()
            if venue:
                venue_aggregate[venue].append(np.log1p(indeg[idx[u]]))
    venue_mean = {k: float(np.mean(v)) for k, v in venue_aggregate.items()}
    global_venue = float(np.mean([x for a in venue_aggregate.values() for x in a])) if venue_aggregate else 0.0

    def x8(node):
        hits = peryear[node]
        return (hits[T] + hits[T-1]) / 2.0

    if novelty:
        novelty.prepare(G)
        nov = novelty.calculate(G, candidate_ids)
    else:
        nov = {}

    rows = []
    for pid in candidate_ids:
        i = idx[pid]
        d = G.nodes[pid]
        counts = peryear[pid]
        c0, c1, c2 = counts[T], counts[T-1], counts[T-2]
        refs = list(G.successors(pid))
        domains = citing_domains[pid]
        domain_total = sum(domains.values())
        entropy = (-sum((count/domain_total)*math.log(count/domain_total)
                        for count in domains.values()) if domain_total else 0.0)
        first_year = d["year"]
        first_1y = counts[first_year] + counts[first_year+1] if first_year+1 <= T else counts[first_year]
        venue = d["venue"].strip().lower()
        value = [
            np.log1p(indeg[i]), np.log1p(outdeg[i]),
            np.log(1e-8 + pr[i]), np.log(1e-8 + citerank[i]),
            authority[i], float(kcore.get(pid, 0)),
            float(c0 - c1), float((c0 + c1) / 2.0),
            float(c0 - 2*c1 + c2),
            float(indeg[i] / max(1, T - first_year + 1)),
            float(first_1y / (indeg[i] + 1)),
            float(core_weight[pid]),
            float(max((pr[idx[v]] for v in refs), default=0.0)),
            float(np.mean([x8(v) for v in refs])) if refs else 0.0,
            float((len(venue_aggregate.get(venue, []))*venue_mean.get(venue, 0.0)
                   + 20*global_venue)/(len(venue_aggregate.get(venue, []))+20)),
            float(entropy),
            float(nov.get(pid, 0.0)),
            float((survey_scores or {}).get(pid, 0.0)),
        ]
        rows.append(dict(paper_id=pid, **dict(zip(FEATURE_COLUMNS, value))))
    data = pd.DataFrame(rows)
    assert data[FEATURE_COLUMNS].notna().all().all(), "NaN in features"
    return data


def read_edge_weights(file):
    if not file:
        return None
    df = pd.read_csv(file)
    if not {"source", "target", "weight"}.issubset(df.columns):
        raise ValueError("F1 weights CSV: source,target,weight")
    return {(str(r.source), str(r.target)): float(r.weight) for r in df.itertuples()}


def read_survey_scores(file):
    if not file:
        return None
    df = pd.read_csv(file)
    if not {"paper_id", "p_survey"}.issubset(df.columns):
        raise ValueError("Survey CSV: paper_id,p_survey")
    return {str(r.paper_id): float(np.clip(r.p_survey, 0, 1)) for r in df.itertuples()}
