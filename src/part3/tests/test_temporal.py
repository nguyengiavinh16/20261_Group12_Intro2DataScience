import networkx as nx
import numpy as np

from f3_future.data import snapshot
from f3_future.labels import make_targets
from f3_future.features import extract_features


def sample_graph():
    G = nx.DiGraph()
    for name, year in [("a", 2013), ("b", 2014), ("c", 2015), ("d", 2016),
                       ("e", 2017), ("f", 2018), ("g", 2020)]:
        G.add_node(name, year=year, title="machine learning example", abstract="machine learning",
                   venue="ML", raw_keywords=[], main_domain="ML")
    G.add_edges_from([("b", "a"), ("c", "a"), ("d", "a"), ("e", "b"),
                      ("f", "b"), ("g", "a")])
    return G


def test_snapshot_no_future():
    G = sample_graph()
    S = snapshot(G, 2015)
    assert set(S) == {"a", "b", "c"}
    assert S.in_degree("a") == 2
    assert not S.has_edge("g", "a")


def test_label_future_growth():
    G = sample_graph()
    a = snapshot(G, 2015)
    b = snapshot(G, 2018)
    labels = make_targets(a, b, 2015, window=3, min_group=2)
    row = labels.set_index("paper_id").loc["a"]
    assert row.c_t == 2
    assert row.c_t3 == 3
    assert np.isclose(row.y_growth, np.log(4) - np.log(3))


def test_graph_features_are_asof():
    G = sample_graph()
    early = snapshot(G, 2015)
    d = extract_features(early, 2015, ["a", "b", "c"])
    row = d.set_index("paper_id").loc["a"]
    assert np.isclose(row.x01, np.log(3))
    assert row.x16 == 0.0  # only one citing domain
