"""Graph loading, cohort/domain assignment and safe historical snapshots."""
import json
import re
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd


DOMAIN_PATTERNS = {
    "NLP": r"\b(nlp|language model|natural language|machine translation|question answering|text generation|bert|tokeniz|speech recognition)\b",
    "CV": r"\b(cv|cvpr|iccv|eccv|computer vision|image|video|object detection|segmentation|point cloud|3d reconstruction|visual recognition)\b",
    "Robotics": r"\b(robot|robotics|navigation|slam|autonomous driving|manipulation|reinforcement control)\b",
    "DataMining": r"\b(kdd|sigmod|vldb|data mining|database|recommender|information retrieval|web search)\b",
    "Theory": r"\b(optimization theory|convergence proof|complexity theory|generalization bound|statistical learning theory)\b",
    "ML": r"\b(ml|neurips|nips|icml|iclr|machine learning|deep learning|neural network|reinforcement learning|representation learning)\b",
}


def _year(value):
    try:
        y = int(float(value))
        return y if 1800 <= y <= 2100 else None
    except (ValueError, TypeError):
        return None


def _keywords(value):
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        decoded = json.loads(value)
        if isinstance(decoded, list):
            return [str(x) for x in decoded]
    except (ValueError, TypeError):
        pass
    return [value]


from pathlib import Path
import pickle
import networkx as nx


def load_graph(graph_file):
    graph_file = Path(graph_file)

    if not graph_file.exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_file}"
        )

    if graph_file.stat().st_size == 0:
        raise ValueError(
            f"Graph file is empty: {graph_file}"
        )

    print(f"[DATA] Reading {graph_file}...")

    if graph_file.suffix.lower() in {".pkl", ".pickle"}:
        # Chỉ đọc pickle từ nguồn đáng tin cậy.
        with open(graph_file, "rb") as f:
            graph = pickle.load(f)

    elif graph_file.suffix.lower() == ".graphml":
        graph = nx.read_graphml(graph_file)

    else:
        raise ValueError(
            f"Unsupported graph format: {graph_file.suffix}"
        )

    if not isinstance(graph, nx.DiGraph):
        raise TypeError(
            "Expected a directed NetworkX graph."
        )

    print(
        f"[DATA] Loaded: "
        f"{graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges"
    )

    return graph


def assign_domains(graph, domain_csv=None):
    """Main domain: CSV mapping preferred; otherwise transparent deterministic heuristic."""
    mapping = {}
    if domain_csv:
        df = pd.read_csv(domain_csv, dtype=str)
        if not {"paper_id", "main_domain"}.issubset(df.columns):
            raise ValueError("domain-map CSV cần cột paper_id,main_domain")
        mapping = dict(zip(df.paper_id, df.main_domain))

    patterns = {key: re.compile(regex, re.IGNORECASE) for key, regex in DOMAIN_PATTERNS.items()}
    counter = Counter()
    for pid, data in graph.nodes(data=True):
        domain = mapping.get(str(pid), "").strip() if mapping else ""
        if not domain:
            keywords = " ".join(data["raw_keywords"])
            venue = data["venue"]
            title = data["title"]
            # priority: raw keywords, then venue, then title; avoid future-dependent fitting.
            for source in (keywords, venue, title):
                for name, pattern in patterns.items():
                    if pattern.search(source):
                        domain = name
                        break
                if domain:
                    break
        data["main_domain"] = domain or "Other"
        counter[data["main_domain"]] += 1
    print(f"[DOMAIN] {dict(counter)}", flush=True)
    return graph


def snapshot(full_graph, T):
    """Edges are observed in citing-paper publication year (approximation)."""
    ids = [n for n, d in full_graph.nodes(data=True) if d["year"] <= T]
    G = full_graph.subgraph(ids).copy()
    # An earlier paper cannot legitimately cite a later published one.
    invalid = [(u, v) for u, v in G.edges() if G.nodes[u]["year"] < G.nodes[v]["year"]]
    G.remove_edges_from(invalid)
    print(f"[SNAPSHOT {T}] nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}, "
          f"dropped_future_reference={len(invalid):,}", flush=True)
    return G


def eligible_papers(G, T, window=3):
    return [n for n, d in G.nodes(data=True) if T-window+1 <= d["year"] <= T]


def cumulative_citations(G):
    return dict(G.in_degree())


def future_counts_for_candidates(full_graph, candidates, future_T):
    """Compute cumulative citations as of future_T without copying a second huge graph.

    A citation edge is dated by source publication year. Invalid source-earlier-
    than-target edges are excluded, consistently with snapshot().
    """
    years = {pid: data["year"] for pid, data in full_graph.nodes(data=True)}
    result = {}
    for pid in candidates:
        target_year = years[pid]
        if target_year > future_T:
            result[pid] = 0
        else:
            result[pid] = sum(1 for src in full_graph.predecessors(pid)
                              if target_year <= years[src] <= future_T)
    return result
