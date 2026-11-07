import numpy as np

from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer


def title_tokens(title):
    return set(title.split())


def exact_jaccard(a, b):
    if not a and not b:
        return 0.0

    return len(a & b) / len(a | b)


def build_minhash(tokens, num_perm=128):
    m = MinHash(num_perm=num_perm)

    for token in sorted(tokens):
        m.update(token.encode("utf-8"))

    return m


class UnionFind:
    def __init__(self, elements):
        self.parent = {x: x for x in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra


def entity_resolution(papers):
    ids = list(papers.keys())

    if not ids:
        return papers, {}

    # 1. TF-IDF cho abstract
    abstracts = [
        papers[pid]["abstract"] or ""
        for pid in ids
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000
    )

    if not any(text.strip() for text in abstracts):
        return papers, {pid: pid for pid in ids}

    try:
        X = vectorizer.fit_transform(abstracts)
    except ValueError:
        return papers, {pid: pid for pid in ids}

    id_to_idx = {
        pid: i for i, pid in enumerate(ids)
    }

    # 2. MinHash LSH
    lsh = MinHashLSH(
        threshold=0.85,
        num_perm=128
    )

    token_sets = {}
    signatures = {}

    for pid in ids:
        tokens = title_tokens(
            papers[pid]["normalized_title"]
        )

        token_sets[pid] = tokens

        if not tokens:
            continue

        mh = build_minhash(tokens)
        signatures[pid] = mh

        lsh.insert(pid, mh)

    # 3. Xác minh ứng viên
    uf = UnionFind(ids)

    checked_pairs = set()
    duplicate_pairs = []

    for pid in ids:
        if pid not in signatures:
            continue

        candidates = lsh.query(signatures[pid])

        for candidate in candidates:
            if candidate == pid:
                continue

            pair = tuple(sorted((pid, candidate)))

            if pair in checked_pairs:
                continue

            checked_pairs.add(pair)

            u, v = pair

            yu = papers[u]["year"]
            yv = papers[v]["year"]

            if yu is None or yv is None:
                continue

            if abs(yu - yv) > 1:
                continue

            jaccard = exact_jaccard(
                token_sets[u],
                token_sets[v]
            )

            if jaccard < 0.85:
                continue

            iu = id_to_idx[u]
            iv = id_to_idx[v]

            # TF-IDF được L2-normalize mặc định.
            # Tích vô hướng = Cosine Similarity.
            cosine = float(
                X[iu].multiply(X[iv]).sum()
            )

            if cosine < 0.90:
                continue

            duplicate_pairs.append(
                (u, v, jaccard, cosine)
            )

            uf.union(u, v)

    # 4. Tạo các nhóm trùng lặp
    clusters = {}

    for pid in ids:
        root = uf.find(pid)
        clusters.setdefault(root, []).append(pid)

    # 5. Chọn bài đại diện
    canonical_map = {}
    merged_papers = {}

    for members in clusters.values():

        def representative_score(pid):
            paper = papers[pid]
            venue = paper["venue"].lower()

            is_preprint = (
                "arxiv" in venue
                or "corr" in venue
            )

            return (
                not is_preprint,
                bool(paper["abstract"]),
                len(paper["abstract"])
            )

        representative = max(
            sorted(members),
            key=representative_score
        )

        for pid in members:
            canonical_map[pid] = representative

        paper = papers[representative].copy()

        # Giữ 6 thuộc tính yêu cầu trên node
        merged_papers[representative] = {
            "paper_id": representative,
            "title": paper["title"],
            "abstract": paper["abstract"],
            "venue": paper["venue"],
            "year": paper["year"],
            "raw_keywords": sorted(set(
                str(keyword)
                for member in members
                for keyword in papers[member]["raw_keywords"]
            ))
        }

    print("Original papers:", len(papers))
    print("Merged papers:", len(merged_papers))
    print("Verified duplicate pairs:", len(duplicate_pairs))

    return merged_papers, canonical_map