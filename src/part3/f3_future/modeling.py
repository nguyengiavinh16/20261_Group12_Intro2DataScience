"""Fit baselines and dual-head LightGBM, choose fusion on validation ONLY."""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from .evaluation import percentile_fuse, ranking_metrics, regression_metrics
from .features import FEATURE_COLUMNS


def _fit_ranker(df, features, n_estimators, seed, min_group):
    # Rows from each query must be contiguous and group sizes sum to len(y).
    ordered = df.sort_values(["group_id", "paper_id"]).reset_index(drop=True)
    usable = ordered.groupby("group_id").filter(
        lambda x: len(x) >= min_group and x.y_grade.nunique() >= 2)
    if usable.empty or usable.group_id.nunique() == 0:
        raise ValueError("Không có cohort có >=10 paper và >=2 mức relevance để train LambdaMART.")
    groups = usable.groupby("group_id", sort=True).size().to_numpy().tolist()
    assert sum(groups) == len(usable)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=[0, 1, 3, 7, 15],
        n_estimators=n_estimators, learning_rate=.03, num_leaves=31,
        min_child_samples=15, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(usable[features], usable.y_grade.astype(int), group=groups, eval_at=[10, 20])
    return model


def _fit_regressor(train, features, n_estimators, seed):
    model = lgb.LGBMRegressor(
        objective="huber", alpha=.9, n_estimators=n_estimators,
        learning_rate=.03, num_leaves=31, subsample=.8, subsample_freq=1,
        colsample_bytree=.8, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(train[features], train.y_growth)
    return model


def prepare_ranking(df, min_group=10):
    return df[df.group_size >= min_group].copy().reset_index(drop=True)


def run_models(train, val, test, output_dir, trees=400, extra_baselines=False, seed=42, min_group=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Xtr, Xv, Xte = [d[FEATURE_COLUMNS] for d in (train, val, test)]
    print("[MODEL] Fitting LightGBM Huber growth head", flush=True)
    growth_model = _fit_regressor(train, FEATURE_COLUMNS, trees, seed)
    print("[MODEL] Fitting LightGBM LambdaMART ranking head", flush=True)
    rank_model = _fit_ranker(train, FEATURE_COLUMNS, trees, seed, min_group)
    for df, X in [(val, Xv), (test, Xte)]:
        df["pred_growth"] = growth_model.predict(X)
        df["pred_rank"] = rank_model.predict(X)

    # Non-ML baselines use exact as-of-T snapshot features.
    baselines = {"in_degree": "x01", "pagerank": "x03", "citerank": "x04",
                 "hits_authority": "x05"}
    for name, feature in baselines.items():
        for df in (val, test):
            df[f"score_{name}"] = df[feature]

    ml_baselines = {
        "ridge": make_pipeline(RobustScaler(), Ridge(alpha=10.0)),
        "elasticnet": make_pipeline(RobustScaler(), ElasticNet(alpha=.05, l1_ratio=.5, max_iter=3000, random_state=seed)),
    }
    if train.y_high_impact.nunique() > 1:
        ml_baselines["logistic_high_impact"] = make_pipeline(
            RobustScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                                random_state=seed))
    else:
        print("[MODEL] Skipping Logistic: only one binary label in Train", flush=True)
    if extra_baselines:
        ml_baselines["random_forest"] = RandomForestRegressor(
            n_estimators=400, max_depth=12, min_samples_leaf=5, n_jobs=4, random_state=seed)
        ml_baselines["extra_trees_classifier"] = ExtraTreesClassifier(
            n_estimators=400, class_weight="balanced_subsample", n_jobs=4,
            random_state=seed)
        # Quadratic SVR: limit training size explicitly, report this in the MD.
        ml_baselines["svr_sampled"] = make_pipeline(
            RobustScaler(), SVR(kernel="rbf", C=2.0, epsilon=.1))

    for name, model in ml_baselines.items():
        print(f"[MODEL] Baseline {name}", flush=True)
        classification = name in {"logistic_high_impact", "extra_trees_classifier"}
        if name == "extra_trees_classifier" and train.y_high_impact.nunique() < 2:
            continue
        fit_df = train
        if name == "svr_sampled" and len(train) > 8000:
            fit_df = train.sample(8000, random_state=seed)
        model.fit(fit_df[FEATURE_COLUMNS], fit_df.y_high_impact if classification else fit_df.y_growth)
        for df in (val, test):
            if classification:
                df[f"score_{name}"] = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
            else:
                df[f"score_{name}"] = model.predict(df[FEATURE_COLUMNS])
        joblib.dump(model, output_dir / f"{name}.joblib")

    # Select fusion coefficient exclusively using validation labels.
    weights = [0.0, .25, .5, .6, .75, 1.0]
    result_by_weight = {}
    for weight in weights:
        val["score_fused"] = percentile_fuse(val, "pred_growth", "pred_rank", weight)
        result_by_weight[weight] = ranking_metrics(
            prepare_ranking(val, min_group), "score_fused", min_group)["ndcg@20"]
    chosen = max(weights, key=lambda w: (-float("inf") if result_by_weight[w] is None else result_by_weight[w], -w))
    print(f"[MODEL] Fusion weight selected on Validation: {chosen}", flush=True)
    for df in (val, test):
        df["score_fused"] = percentile_fuse(df, "pred_growth", "pred_rank", chosen)

    report = {"chosen_fusion_weight": chosen, "validation_fusion_ndcg20": result_by_weight,
              "validation": {}, "test": {}}
    score_cols = [f"score_{name}" for name in baselines] + [
        f"score_{name}" for name in ml_baselines if f"score_{name}" in test.columns] + [
        "pred_growth", "pred_rank", "score_fused"]
    for split, df in [("validation", val), ("test", test)]:
        ranked = prepare_ranking(df, min_group)
        for col in score_cols:
            report[split][col] = ranking_metrics(ranked, col, min_group)
        report[split]["growth_regression"] = regression_metrics(df.y_growth, df.pred_growth)
    diagnostics = []
    for split, df in [("train", train), ("validation", val), ("test", test)]:
        for group_id, group in df.groupby("group_id", sort=True):
            diagnostics.append({
                "split": split,
                "group_id": group_id,
                "n_papers": len(group),
                "high_impact_positives": int(group.y_high_impact.sum()),
                "future_zero_citations": int((group.c_t3 == 0).sum()),
                "n_relevance_levels": int(group.y_grade.nunique()),
                "eligible": bool(len(group) >= min_group),
            })
    pd.DataFrame(diagnostics).to_csv(output_dir / "cohort_diagnostics.csv", index=False)

    joblib.dump(growth_model, output_dir / "lightgbm_growth.joblib")
    joblib.dump(rank_model, output_dir / "lightgbm_ranker.joblib")
    # Save feature schema and configuration for interpretation.
    (output_dir / "models_config.json").write_text(json.dumps({
        "feature_columns": FEATURE_COLUMNS, "fusion_weight": chosen,
        "training_cohorts": sorted(train.group_id.unique().tolist()),
    }, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    val.to_csv(output_dir / "predictions_validation.csv", index=False)
    test.to_csv(output_dir / "predictions_test.csv", index=False)
    return report
