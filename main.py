"""Main pipeline: JSONL -> entity resolution -> G_cite -> G_flow -> saved graphs.

Run:
    python main.py --papers data/raw/papers.jsonl --references data/raw/references.jsonl
    python main.py --mode similarity --format graphml
    python main.py --cite data/processed/G_cite.pkl --mode fast
"""
import argparse
import json
import time
from pathlib import Path

import networkx as nx

from src.process_data.graph_builder import (
    build_citation_graph,
    build_flow_graph,
    read_graph,
    save_graph_pickle,
    save_graphml_safe,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build citation and knowledge-flow graphs")
    parser.add_argument("--papers", default="data/raw/papers_final.jsonl")
    parser.add_argument("--references", default="data/raw/references_final.jsonl")
    parser.add_argument("--cite", default=None,
                        help="Use an existing G_cite (.pkl/.graphml), skipping loading and deduplication")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--mode", choices=["fast", "similarity"], default="similarity")
    parser.add_argument("--format", choices=["pkl", "graphml", "both"], default="pkl")
    return parser.parse_args()


def save_graph(graph, out_dir, stem, fmt):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if fmt in ("pkl", "both"):
        path = out_dir / f"{stem}.pkl"
        save_graph_pickle(graph, path)
        paths.append(str(path))
    if fmt in ("graphml", "both"):
        path = out_dir / f"{stem}.graphml"
        # Converts list/dict (e.g. raw_keywords) to JSON strings before writing.
        save_graphml_safe(graph, path)
        paths.append(str(path))
    return paths


def main():
    args = parse_args()
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cite:
        print(f"STEP 1: Loading existing G_cite: {args.cite}", flush=True)
        G_cite = read_graph(args.cite)
    else:
        from src.process_data.data_loader import load_papers, load_references
        from src.process_data.entity_resolution import entity_resolution

        print("STEP 1: Loading data", flush=True)
        papers = load_papers(args.papers)
        references = load_references(args.references)
        print(f"Papers: {len(papers):,}; reference records: {len(references):,}", flush=True)

        print("STEP 2: Entity resolution", flush=True)
        merged_papers, canonical_map = entity_resolution(papers)
        print(f"Merged papers: {len(merged_papers):,}", flush=True)

        print("STEP 3: Building G_cite", flush=True)
        G_cite = build_citation_graph(merged_papers, references, canonical_map)
        del papers, references, merged_papers, canonical_map

    print(f"G_cite: {G_cite.number_of_nodes():,} nodes, "
          f"{G_cite.number_of_edges():,} edges", flush=True)

    print("STEP 4: Building G_flow", flush=True)
    G_flow, flow_stats = build_flow_graph(
        G_cite, mode=args.mode, return_stats=True, verbose=True
    )
    assert nx.is_directed_acyclic_graph(G_flow), "G_flow must be a DAG"
    print(f"G_flow: {G_flow.number_of_nodes():,} nodes, "
          f"{G_flow.number_of_edges():,} edges", flush=True)

    print("STEP 5: Saving graphs", flush=True)
    paths = save_graph(G_cite, out_dir, "G_cite", args.format)
    paths += save_graph(G_flow, out_dir, "G_flow", args.format)
    for path in paths:
        print(f"Saved: {path}", flush=True)

    stats = {
        "mode": args.mode,
        "format": args.format,
        "G_cite_nodes": G_cite.number_of_nodes(),
        "G_cite_edges": G_cite.number_of_edges(),
        "G_flow_nodes": G_flow.number_of_nodes(),
        "G_flow_edges": G_flow.number_of_edges(),
        "G_flow_is_DAG": True,
        "flow_build_stats": flow_stats,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    stats_path = out_dir / "graph_build_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Saved: {stats_path}", flush=True)
    print(f"DONE in {stats['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
