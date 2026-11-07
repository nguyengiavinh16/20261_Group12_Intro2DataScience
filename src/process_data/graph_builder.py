"""Build citation and time-respecting knowledge-flow graphs.

Optimized for large graphs with many edges and sparse same-year cycles.

Two cycle-breaking policies:
    mode="similarity": choose the least TF-IDF-cosine-similar edge in each
        detected cycle, searching only within cyclic SCCs. Faithful to the
        project's intended rule (apart from fitting IDF over cyclic abstracts).
    mode="fast": order nodes within each cyclic SCC by net degree and keep
        forward edges only. Much faster worst-case, but approximate and may
        remove additional edges without using TF-IDF cosine.

Both policies preserve the original node set, only reverse time-valid citation
edges, and guarantee a DAG. No GPU is needed.
"""

from __future__ import annotations

import json
import math
import pickle
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Optional

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _year(value: Any) -> Optional[int]:
    """Convert integer or GraphML year attribute to a valid year."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            return None
        result = int(number)
        return result if result > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def build_citation_graph(
    papers: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Iterable[str]],
    canonical_map: Mapping[str, str],
) -> nx.DiGraph:
    """Create G_cite with edges (citing_paper -> cited_paper)."""
    graph = nx.DiGraph()
    graph.add_nodes_from((pid, dict(attrs)) for pid, attrs in papers.items())
    for raw_source, raw_targets in references.items():
        source = canonical_map.get(raw_source, raw_source)
        if source not in graph:
            continue
        for raw_target in raw_targets:
            target = canonical_map.get(raw_target, raw_target)
            if target in graph and source != target:
                graph.add_edge(source, target)
    return graph


def calculate_citerank(G: nx.DiGraph, current_year: int = 2026,
                       tau: float = 5.0) -> Dict[str, float]:
    """Personalized PageRank with exponential age-decay teleportation."""
    if G.number_of_nodes() == 0:
        return {}
    if tau <= 0:
        raise ValueError("tau must be positive")
    weights = {
        node: (math.exp(-max(0, current_year - yr) / tau)
               if (yr := _year(attrs.get("year"))) is not None else 0.0)
        for node, attrs in G.nodes(data=True)
    }
    total = sum(weights.values())
    if total <= 0:
        weights = {node: 1.0 / len(G) for node in G}
    else:
        weights = {node: weight / total for node, weight in weights.items()}
    return nx.pagerank(G, alpha=0.85, personalization=weights,
                       dangling=weights)


def build_flow_graph(
    G_cite: nx.DiGraph,
    mode: str = "fast",
    *,
    return_stats: bool = False,
    verbose: bool = True,
    max_features: int = 50_000,
):
    """Return the time-respecting acyclic graph G_flow.

    Args:
        G_cite: citation graph u_new -> v_old, with node attribute ``year``.
        mode: ``fast`` (greedy per-SCC topological order) or ``similarity``
            (least TF-IDF-cosine edge on each found cycle, per cyclic SCC).
        return_stats: return (graph, dictionary_of_diagnostics) if true.
        verbose: print progress at logical boundaries.
        max_features: vocabulary size in similarity mode.

    Nodes with missing/invalid years remain but their incident edges are
    excluded from G_flow, because chronology cannot be validated.
    """
    if mode not in {"fast", "similarity"}:
        raise ValueError("mode must be 'fast' or 'similarity'")
    if not G_cite.is_directed():
        raise ValueError("G_cite must be directed")

    def report(message):
        if verbose:
            print(message, flush=True)

    total_start = perf_counter()
    stats: Dict[str, Any] = {"mode": mode, "input_nodes": G_cite.number_of_nodes(),
                             "input_edges": G_cite.number_of_edges()}
    years = {node: _year(attrs.get("year")) for node, attrs in G_cite.nodes(data=True)}

    # Preserve every node (including nodes with unknown years).
    G_flow = nx.DiGraph()
    G_flow.add_nodes_from(G_cite.nodes(data=True))
    same_year_graph = nx.DiGraph()
    reverse_time = 0
    missing_time = 0
    same_year_edges = 0
    self_loops = 0

    # Construct the graph and the (much smaller) same-year subgraph in one pass.
    for citing, cited in G_cite.edges():
        yc, yr = years[citing], years[cited]
        if citing == cited:
            self_loops += 1
        elif yc is None or yr is None:
            missing_time += 1
        elif yr > yc:
            reverse_time += 1
        else:
            G_flow.add_edge(cited, citing)
            if yc == yr:
                same_year_graph.add_edge(cited, citing)
                same_year_edges += 1

    stats.update(removed_reverse_time=reverse_time,
                 removed_missing_year=missing_time,
                 removed_self_loops=self_loops,
                 same_year_edges=same_year_edges,
                 initial_flow_edges=G_flow.number_of_edges())
    report(f"[1/4] Temporal filter: {G_flow.number_of_edges():,} edges kept; "
           f"{reverse_time:,} invalid-time edges; {same_year_edges:,} same-year edges")
    t1 = perf_counter()

    # Every remaining cycle is entirely within a single publication year.
    # Hence only the SCCs of the same-year graph can contain cycles.
    cyclic_components = [part for part in nx.strongly_connected_components(same_year_graph)
                         if len(part) > 1]
    stats["cyclic_scc_count"] = len(cyclic_components)
    stats["largest_cyclic_scc"] = max((len(p) for p in cyclic_components), default=0)
    report(f"[2/4] Cyclic SCCs: {len(cyclic_components):,}; "
           f"largest SCC: {stats['largest_cyclic_scc']:,} nodes")
    t2 = perf_counter()

    X = None
    index: Dict[Any, int] = {}
    missing_abstract = set()
    if cyclic_components and mode == "similarity":
        cyclic_nodes = sorted(set().union(*cyclic_components), key=str)
        abstracts = []
        for node in cyclic_nodes:
            raw = G_flow.nodes[node].get("abstract")
            abstract = raw.strip() if isinstance(raw, str) else ""
            if not abstract:
                missing_abstract.add(node)
            abstracts.append(abstract)
        index = {node: i for i, node in enumerate(cyclic_nodes)}
        if any(abstracts):
            try:
                vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
                X = vectorizer.fit_transform(abstracts)
            except ValueError:  # no valid terms in corpus
                pass
        report(f"[3/4] TF-IDF computed for {len(cyclic_nodes):,} cyclic nodes "
               f"(missing abstracts: {len(missing_abstract):,})")
    else:
        report("[3/4] TF-IDF skipped (fast mode or no cyclic SCCs)")

    @lru_cache(maxsize=200_000)
    def similarity(u, v):
        if X is None or u in missing_abstract or v in missing_abstract:
            return None
        return float(X[index[u]].multiply(X[index[v]]).sum())

    def priority(edge):
        u, v = edge
        sim = similarity(u, v)
        # Missing similarity receives LOW removal priority: remove such edges
        # only when all candidate cycle edges lack valid similarities.
        if sim is None:
            return (1, 0.0, str(u), str(v))
        return (0, sim, str(u), str(v))

    removed_cycle = 0
    for component in cyclic_components:
        H = same_year_graph.subgraph(component)
        if mode == "fast":
            # Greedy order: likely sources (large out-degree) before sinks.
            # Every retained edge must point strictly forward in this order.
            ordering = sorted(component, key=lambda n: (
                H.in_degree(n) - H.out_degree(n), str(n)))
            rank = {node: i for i, node in enumerate(ordering)}
            for u, v in H.edges():
                if rank[u] >= rank[v]:
                    G_flow.remove_edge(u, v)
                    removed_cycle += 1
        else:
            # Work on a small SCC, not the full ~million-edge graph.
            local = H.copy()
            while True:
                try:
                    cycle = nx.find_cycle(local)
                except nx.NetworkXNoCycle:
                    break
                target = min(cycle, key=priority)
                local.remove_edge(*target)
                G_flow.remove_edge(*target)
                removed_cycle += 1

    stats["removed_cycle_edges"] = removed_cycle
    stats["output_nodes"] = G_flow.number_of_nodes()
    stats["output_edges"] = G_flow.number_of_edges()
    report(f"[4/4] Cycle breaking: removed {removed_cycle:,} edges")
    del same_year_graph
    del cyclic_components

    if not nx.is_directed_acyclic_graph(G_flow):
        raise AssertionError("Unexpected cycle in G_flow")
    for u, v in G_flow.edges():
        # Already verified at construction, check rapidly during development.
        if years[u] is None or years[v] is None or years[u] > years[v]:
            raise AssertionError("Found invalid temporal edge")
    stats["seconds_temporal"] = round(t1 - total_start, 3)
    stats["seconds_scc"] = round(t2 - t1, 3)
    stats["seconds_total"] = round(perf_counter() - total_start, 3)
    report(f"G_flow READY: {G_flow.number_of_nodes():,} nodes / "
           f"{G_flow.number_of_edges():,} edges; DAG=True; "
           f"{stats['seconds_total']:.2f}s")
    if return_stats:
        return G_flow, stats
    return G_flow


def save_graphml_safe(G: nx.DiGraph, destination: str | Path) -> None:
    """Write GraphML without copying millions of edges just for serialization.

    List/dict attributes (such as raw_keywords) are temporarily serialized
    to JSON strings; original in-memory graph attributes are restored later.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    changes = []
    for _, attrs in G.nodes(data=True):
        for key, value in list(attrs.items()):
            if value is None or isinstance(value, (list, dict, tuple, set)):
                changes.append((attrs, key, value))
                if value is None:
                    attrs[key] = ""
                elif isinstance(value, set):
                    attrs[key] = json.dumps(sorted(value, key=str), ensure_ascii=False)
                else:
                    attrs[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, np.generic):
                changes.append((attrs, key, value))
                attrs[key] = value.item()
    try:
        nx.write_graphml(G, destination, prettyprint=False)
    finally:
        for attrs, key, value in changes:
            attrs[key] = value


def save_graph_pickle(G: nx.DiGraph, destination: str | Path) -> None:
    """Faster local cache; only unpickle files that you created/trust."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        pickle.dump(G, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_graph(path: str | Path) -> nx.DiGraph:
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        # Warning: Python pickle is unsafe for untrusted or downloaded files.
        with path.open("rb") as file:
            result = pickle.load(file)
    else:
        result = nx.read_graphml(path)
    if not isinstance(result, nx.DiGraph):
        raise TypeError("Expected nx.DiGraph in saved graph")
    return result
