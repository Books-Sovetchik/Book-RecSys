import os
import sys
import json
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

class RecommendUsingGraph:
    def __init__(self, graph_path, embds_path):
        self.graph = self.load_graph(graph_path)
        self.embds = np.load(embds_path, allow_pickle=True)

    def load_graph(self, graph_path):
        """Loads a graph from a JSON file and returns a NetworkX graph."""
        with open(graph_path, "r") as f:
            graph_data = json.load(f)

        Graph = json_graph.node_link_graph(graph_data) 
        return Graph

    def find_neighbors_title(self, title):
        """Find books directly connected to the given title in the graph."""
        matching_nodes = [node for node in self.graph.nodes if node.startswith(f"{title} (")]
        all_neighbors = set()  

        for node in matching_nodes:
            neighbors = list(self.graph.neighbors(node))
            all_neighbors.update(neighbors)

        return [neighbor.rsplit(" (", 1)[0] for neighbor in all_neighbors]
    
    def recommend_graph(self, title, n=10):
        """Recommend top n graph-neighbor books using embedding similarity."""
        neighbors = self.find_neighbors_title(title)

        titles = self.embds["titles"]
        embeddings = self.embds["embeddings"]
        title_to_embedding = {
            title: np.array(embd, dtype=np.float32)
            for title, embd in zip(titles, embeddings)
        }

        title_embedding = title_to_embedding.get(title)
        if title_embedding is None:
            return []

        norm_title = np.linalg.norm(title_embedding)
        neighbor_scores = {}

        for neighbor in neighbors:
            print(neighbor)
            if neighbor == title:
                continue
            neighbor_embedding = title_to_embedding.get(neighbor)
            if neighbor_embedding is None:
                continue

            norm_neighbor = np.linalg.norm(neighbor_embedding)
            similarity = (
                np.dot(title_embedding, neighbor_embedding) / (norm_title * norm_neighbor)
                if norm_title and norm_neighbor else 0.0
            )

            neighbor_scores[neighbor] = similarity

        sorted_neighbors = sorted(neighbor_scores.items(), key=lambda x: -x[1])[:n]
        res = []
        for book, _ in sorted_neighbors:
            res.append(book)
        return res