import numpy as np

from iiot_msresstgat.graph import build_graph, swat_stage


def test_swat_stage_parser():
    assert swat_stage("FIT301") == 3
    assert swat_stage("unknown") is None


def test_graph_uses_physical_stage_constraint():
    values = np.arange(30, dtype=float).reshape(10, 3)
    graph = build_graph(values, ["FIT101", "FIT201", "FIT601"], 0.9, True)
    edges = set(map(tuple, graph.T.tolist()))
    assert (0, 1) in edges
    assert (0, 2) not in edges

