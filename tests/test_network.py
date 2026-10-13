import cv2
import numpy as np

from statemachine_pipeline.network import NetworkEdge, NetworkGraph, NetworkNode, validate_network_graph
from statemachine_pipeline.network_cv import detect_network
from generate_network_diagrams import generate


def test_network_contract_rejects_dangling_edge():
    graph = NetworkGraph(
        "network-1.0", "x", (NetworkNode("N0", "A", {"x": 10, "y": 10}, 20),),
        (NetworkEdge("E0", "N0", "missing"),),
    )
    assert "$.edges[0].target" in validate_network_graph(graph)[0]


def test_network_cv_detects_circle_components_and_connectors(tmp_path):
    image = np.full((360, 560), 255, np.uint8)
    for center in ((110, 180), (280, 100), (450, 220)):
        cv2.circle(image, center, 38, 0, 4)
    cv2.line(image, (140, 160), (250, 115), 0, 4)
    cv2.line(image, (310, 125), (420, 200), 0, 4)
    path = tmp_path / "network.png"
    assert cv2.imwrite(str(path), image)

    graph = detect_network(path)

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert not validate_network_graph(graph)


def test_network_generator_writes_images_and_ground_truth(tmp_path):
    generate(tmp_path, count=2, seed=11)
    images = sorted(tmp_path.glob("*.png"))
    labels = sorted(tmp_path.glob("*.json"))
    assert len(images) == len(labels) == 2
    for label in labels:
        graph = NetworkGraph.from_dict(__import__("json").loads(label.read_text()))
        assert graph.metadata["source"] == "synthetic"
        assert not validate_network_graph(graph)
