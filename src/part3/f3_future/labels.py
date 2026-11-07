"""Future 3-year targets and cohort-normalized relevance labels."""
import numpy as np
import pandas as pd


def make_targets(G_snapshot, G_future, T, window=3, min_group=10):
    candidates = [pid for pid, data in G_snapshot.nodes(data=True)
                  if T-window+1 <= data["year"] <= T]
    c_now = dict(G_snapshot.in_degree())
    c_fut = G_future if isinstance(G_future, dict) else dict(G_future.in_degree())
    if not candidates:
        raise ValueError(f"Không có bài báo ứng viên tại T={T}; kiểm tra khoảng năm.")
    rows = []
    for pid in candidates:
        attrs = G_snapshot.nodes[pid]
        a, b = c_now.get(pid, 0), c_fut.get(pid, 0)
        rows.append({"paper_id": pid, "publication_year": attrs["year"],
                     "main_domain": attrs["main_domain"], "snapshot_year": T,
                     "c_t": a, "c_t3": b,
                     "y_growth": float(np.log1p(b) - np.log1p(a))})
    df = pd.DataFrame(rows)
    df["group_id"] = df["publication_year"].astype(str) + "|" + df["main_domain"]
    df["group_size"] = df.groupby("group_id")["paper_id"].transform("size")
    # Mid-rank; do not arbitrarily label zero-citation ties as Top 5%.
    df["y_pct"] = df.groupby("group_id")["c_t3"].rank(method="average", pct=True)
    bins = [-np.inf, 0.50, 0.75, 0.90, 0.95, np.inf]
    df["y_grade"] = pd.cut(df["y_pct"], bins=bins, labels=False).astype(int)
    df["y_high_impact"] = (df["y_pct"] > 0.95).astype(int)
    df["eligible_rank_group"] = df["group_size"] >= min_group
    eligible = df[df.eligible_rank_group]
    print(f"[TARGET T={T}] total={len(df):,}; eligible={len(eligible):,}; "
          f"groups={eligible.group_id.nunique()}; positive={int(eligible.y_high_impact.sum())}", flush=True)
    return df
