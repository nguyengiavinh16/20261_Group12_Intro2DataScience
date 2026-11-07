"""Group-aware ranking and regression evaluation; never used for model fitting."""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _ndcg(y, scores, k):
    n = min(k, len(y))
    if n == 0:
        return np.nan
    order = np.argsort(-np.asarray(scores), kind="stable")
    ideal = np.argsort(-np.asarray(y), kind="stable")
    gain = lambda rel: np.power(2.0, np.asarray(rel, dtype=float)) - 1.0
    discount = np.log2(np.arange(2, n+2))
    d = np.sum(gain(np.asarray(y)[order[:n]]) / discount)
    best = np.sum(gain(np.asarray(y)[ideal[:n]]) / discount)
    return float(d/best) if best > 0 else np.nan


def _ap(y, scores, k):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    rel = y[order]
    numer = sum(rel[i] * (rel[:i+1].sum() / (i+1)) for i in range(len(rel)))
    return float(numer / min(int(y.sum()), k))


def _top_precision(y, scores):
    y = np.asarray(y, dtype=int)
    k = max(1, int(np.ceil(len(y)*.05)))
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(np.mean(y[order]))


def ranking_metrics(df, score_column, min_group=10):
    metrics = []
    for _, group in df.groupby("group_id", sort=True):
        if len(group) < min_group:
            continue
        scores = group[score_column].to_numpy()
        label = group.y_high_impact.to_numpy()
        graded = group.y_grade.to_numpy()
        truth = group.c_t3.to_numpy()
        rho = spearmanr(truth, scores).statistic if (len(np.unique(truth)) > 1
                    and len(np.unique(scores)) > 1) else np.nan
        tau = kendalltau(truth, scores).statistic if (len(np.unique(truth)) > 1
                    and len(np.unique(scores)) > 1) else np.nan
        metrics.append({"ndcg@10": _ndcg(graded, scores, 10),
                        "ndcg@20": _ndcg(graded, scores, 20),
                        "map@20": _ap(label, scores, 20),
                        "precision@top5%": _top_precision(label, scores),
                        "spearman": rho, "kendall": tau})
    if not metrics:
        raise ValueError("Không có cohort đủ lớn để tính ranking metrics.")
    table = pd.DataFrame(metrics)
    result = {key: (float(table[key].mean()) if table[key].notna().any() else None)
              for key in table.columns}
    result["ranking_groups"] = len(table)
    result["groups_with_positive"] = int(table["map@20"].notna().sum())
    return result


def regression_metrics(truth, pred):
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    residual = np.abs(truth-pred)
    huber_delta = 1.0
    huber = np.where(residual <= huber_delta, 0.5*residual**2,
                     huber_delta*(residual-0.5*huber_delta))
    return {"rmse": float(np.sqrt(mean_squared_error(truth, pred))),
            "mae": float(mean_absolute_error(truth, pred)),
            "huber_loss_delta1": float(np.mean(huber))}


def percentile_fuse(df, regression_col, ranking_col, weight):
    """Within-query percentile scores, no target labels used."""
    r1 = df.groupby("group_id")[ranking_col].rank(method="average", pct=True)
    r2 = df.groupby("group_id")[regression_col].rank(method="average", pct=True)
    return weight * r1 + (1-weight) * r2
