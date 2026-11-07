"""End-to-end F3 Future High-Impact. Run from project root."""
import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from f3_future.data import load_graph, assign_domains, snapshot, eligible_papers, future_counts_for_candidates
from f3_future.labels import make_targets
from f3_future.features import (NoveltyModel, FEATURE_COLUMNS, extract_features,
                                 read_edge_weights, read_survey_scores)
from f3_future.modeling import run_models


def build_split(G_full, T, horizon, candidate_window, novelty, output_dir,
                min_group, f1_weights, survey_scores):
    G_now = snapshot(G_full, T)
    candidates = eligible_papers(G_now, T, candidate_window)
    future_counts = future_counts_for_candidates(G_full, candidates, T+horizon)
    labels = make_targets(G_now, future_counts, T, candidate_window, min_group)
    features = extract_features(G_now, T, candidates, novelty=novelty,
                                f1_weights=f1_weights, survey_scores=survey_scores)
    dataset = labels.merge(features, on="paper_id", how="inner", validate="one_to_one")
    dataset[FEATURE_COLUMNS] = dataset[FEATURE_COLUMNS].astype("float32")
    dataset.to_csv(Path(output_dir) / f"dataset_T{T}.csv", index=False)
    print(f"[SPLIT {T}] Saved {len(dataset):,} rows", flush=True)
    del G_now
    gc.collect()
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Strict-temporal Future High-Impact F3")
    parser.add_argument("--graph", default="data/processed/G_cite.pkl")
    parser.add_argument("--output", default="outputs/f3_future")
    parser.add_argument("--domain-map", default=None, help="CSV paper_id,main_domain")
    parser.add_argument("--f1-weights", default=None, help="CSV source,target,weight")
    parser.add_argument("--survey-scores", default=None, help="CSV paper_id,p_survey")
    parser.add_argument("--train-T", type=int, default=2015)
    parser.add_argument("--val-T", type=int, default=2018)
    parser.add_argument("--test-T", type=int, default=2021)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--candidate-window", type=int, default=3)
    parser.add_argument("--min-group", type=int, default=10)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--disable-novelty", action="store_true")
    parser.add_argument("--extra-baselines", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.horizon <= 0 or args.candidate_window <= 0 or args.min_group < 2:
        parser.error("horizon, candidate-window > 0 và min-group >= 2")
    if args.train_T + args.horizon > args.val_T:
        parser.error("Y_train = train_T+horizon phải <= val_T")
    if args.val_T + args.horizon > args.test_T:
        parser.error("Y_val = val_T+horizon phải <= test_T")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    G = load_graph(args.graph)
    G = assign_domains(G, args.domain_map)
    max_year = max(d["year"] for _, d in G.nodes(data=True))
    if args.test_T + args.horizon > max_year:
        parser.error(f"Những bài báo có năm xuất bản mới nhất {max_year}; "
                     f"cần dữ liệu đến {args.test_T + args.horizon} cho Test label.")
    print("[PROTOCOL] Training labels known by Validation snapshot; "
          "Validation labels known by Test snapshot", flush=True)

    train_graph = snapshot(G, args.train_T)
    novelty = NoveltyModel(enabled=not args.disable_novelty).fit(train_graph)
    del train_graph
    gc.collect()
    f1_weights = read_edge_weights(args.f1_weights)
    survey_scores = read_survey_scores(args.survey_scores)
    result = {}
    for name, T in [("train", args.train_T), ("validation", args.val_T), ("test", args.test_T)]:
        print(f"\n======== {name.upper()} T={T} ========", flush=True)
        result[name] = build_split(G, T, args.horizon, args.candidate_window,
                                   novelty, output, args.min_group, f1_weights,
                                   survey_scores)
    del G, novelty
    gc.collect()
    report = run_models(result["train"], result["validation"], result["test"],
                        output, trees=args.trees, extra_baselines=args.extra_baselines,
                        seed=args.seed, min_group=args.min_group)
    config = vars(args)
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\n=== TEST BENCHMARK (macro across cohorts) ===", flush=True)
    for name, value in report["test"].items():
        print(name, value, flush=True)
    print(f"[DONE] Saved outputs at: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
