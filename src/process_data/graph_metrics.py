import networkx as nx
import scipy.sparse as sp


def calculate_coupling_and_cocitation(G):
    nodes = list(G.nodes())

    A = nx.to_scipy_sparse_array(
        G,
        nodelist=nodes,
        dtype=float,
        format="csr"
    )

    bibliographic_coupling = A @ A.T

    cocitation = A.T @ A

    return (
        nodes,
        bibliographic_coupling,
        cocitation
    )