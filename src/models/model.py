import numpy as np
from .modules.LSTM_model.lstm_model import UserContextModel
from .modules.book_description_embedding_similarity.embeddings_similarity import BookDescriptionEmbeddingSimilarity
from .modules.book_graph_descriptions.graph_recommender import RecommendUsingGraph
from .modules.sequences.sequence_recommender import SequenceRecommender

import torch

class Bibliotekar():
    def __init__(self, fs_embds, ss_embds, graph_path, sequences_path, leha, model_path):
        fs = np.load(fs_embds, allow_pickle=True)
        ss = np.load(ss_embds, allow_pickle=True)


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embds = {
            "titles": np.concatenate((fs["titles"], ss["titles"]), axis=0),
            "embeddings": np.concatenate((fs["embeddings"], ss["embeddings"]), axis=0)
        }
        self.size = len(self.embds["titles"])
        self.model = UserContextModel(self.embds, model_path, self.device)
        self.graph = RecommendUsingGraph(graph_path, fs_embds)
        self.seq = SequenceRecommender(leha, sequence_path=sequences_path)

    def predict(self, last_book, k=10):
        return self.model.predict_last(last_book, self.embds, top_k=k)["titles"]

    def predict_context(self, last_books, last_book, last_book_title, dataset_ver="fs", k = 10):
        top = []
        if last_book_title != "":
            top += (self.seq.recommend_seq(last_book_title, k))
            top += (self.graph.recommend_graph(last_book_title, k))
        
        first_iter = self.model.predict_books(last_books, self.embds, top_k=self.size//10)
        second_iter = self.model.predict_last(last_book, first_iter, top_k=k)

        return top + second_iter["titles"]