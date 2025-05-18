import numpy as np
from modules.LSTM_model.lstm_model import UserContextModel
from modules.book_description_embedding_similarity.embeddings_similarity import BookDescriptionEmbeddingSimilarity
from modules.book_graph_descriptions.graph_recommender import GraphRecommender
from modules.sequences.sequence_recommender import SequenceRecommender

class Bibliotear():
    def __init__(self, embds_path, model_path, device):
        self.embds = np.load(embds_path)
        self.size = len(self.embds["titles"])
        self.model = UserContextModel(embds_path, model_path, device)
        self.gbe = BookDescriptionEmbeddingSimilarity(embds_path)
        self.graph = GraphRecommender(embds_path)
        self.seq = SequenceRecommender(embds_path)

    def predict(self, last_book, k=10):
        return self.model.predict_last(last_book, self.embds, k=k)

    def predict(self, last_books, last_book, k = 10):
        first_iter = self.model.predict_books(last_books, last_book, self.embds, k=self.size/10)
        return self.model.predict_last(last_book, first_iter, k=k)

