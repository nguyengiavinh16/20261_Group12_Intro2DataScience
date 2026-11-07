"""Create a small fictional publication network with valid timestamps for smoke tests."""
import argparse
import random
from pathlib import Path

import networkx as nx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/demo_G_cite.graphml")
    p.add_argument("--papers-per-year", type=int, default=90)
    args = p.parse_args()
    rng = random.Random(42)
    G = nx.DiGraph()
    domains = ["CV", "NLP", "ML"]
    past = []
    for year in range(2010, 2025):
        batch = []
        for i in range(args.papers_per_year):
            pid = f"demo_{year}_{i:04d}"
            dom = domains[i % len(domains)]
            title = f"{dom} neural framework {i % 17} learning optimization {year}"
            G.add_node(pid, paper_id=pid, title=title,
                       abstract=(f"We propose a {dom} network for representation learning "
                                 f"and improve generalization with optimization approach {i % 9}."),
                       venue=dom, year=year, raw_keywords=f'["{dom}"]')
            batch.append(pid)
            if past:
                choices = rng.sample(past, min(6, len(past)))
                for target in choices:
                    G.add_edge(pid, target)
        past.extend(batch)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, output)
    print(f"Saved synthetic demo: {output} ({len(G)} nodes, {len(G.edges())} edges)")


if __name__ == "__main__":
    main()
