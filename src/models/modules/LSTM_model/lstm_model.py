import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_dim = 768
output_dim = 768
hidden_dim = 128
num_layers = 3
dropout = 0.2


class Rekomendatel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out1, (h_n1, c_n1) = self.lstm1(x)
        lstm_out2, (h_n2, c_n2) = self.lstm2(lstm_out1)
        last_hidden_state = h_n2[-1]
        out = self.dropout(last_hidden_state)
        out = self.fc(out)
        return out


class UserContextModel():
    def __init__(self, data_path, model_path, device):
        self.model = Rekomendatel().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.device = device

    def predict(self, user_prev):
        user_prev = torch.from_numpy(user_prev).float().to(self.device)
        return self.model(user_prev)
    
    def predict_books(self, user_prev, dataset, top_k=10):
        self.dataset = dataset
        self.tittles, self.embeddings = self.dataset["titles"], self.dataset["embeddings"]  

        user_vec = self.predict(user_prev)
        user_vec = F.normalize(user_vec, dim=-1)

        book_embeds = torch.from_numpy(self.embeddings).to(self.device)
        book_embeds = F.normalize(book_embeds, dim=-1)

        similarities = torch.matmul(book_embeds, user_vec.T).squeeze()
        top_indices = torch.topk(similarities, k=top_k).indices.cpu().numpy()

        recommended_titles = [str(self.titles[i]) for i in top_indices]
        recommended_embds = [self.embeddings[i] for i in top_indices]

        return {
            "titles": recommended_titles,
            "embeddings": recommended_embds
        }
        
    def predict_last(self, embd, dataset, top_k=10):
        self.dataset = dataset

        self.titles, self.embeddings = self.dataset["titles"], self.dataset["embeddings"]  

        user_vec = torch.tensor(embd)
        user_vec = F.normalize(user_vec, dim=-1)

        book_embeds = torch.from_numpy(self.embeddings).to(self.device)
        book_embeds = F.normalize(book_embeds, dim=-1)

        similarities = torch.matmul(book_embeds, user_vec.T).squeeze()
        top_indices = torch.topk(similarities, k=top_k).indices.cpu().numpy()

        recommended_titles = [str(self.titles[i]) for i in top_indices]
        recommended_embds = [self.embeddings[i] for i in top_indices]

        return {
            "titles": recommended_titles,
            "embeddings": recommended_embds
        }