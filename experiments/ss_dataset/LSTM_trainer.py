#This file is inteded to train Rekomendatel model


import torch
from torch import nn
from torch import functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import pandas as pd
import numpy as np

import os, sys


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from models.modules.LSTM_model.lstm_model import UserContextModel, Rekomendatel



PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
PATH_BOOKS = os.path.join(DATA_PATH, "raw_data", "kaggle_second_sem", "books_data.csv")
PATH_RATINGS = os.path.join(DATA_PATH, "raw_data", "kaggle_second_sem", "books_rating.csv")
PATH_EMBDS = os.path.join(DATA_PATH, "embeddings", "ss_embds.npz")
df_books = pd.read_csv(PATH_BOOKS)
df_ratings = pd.read_csv(PATH_RATINGS)
embds_npz = np.load(PATH_EMBDS, allow_pickle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


titles, book_embds  = embds_npz["titles"], embds_npz["embeddings"]

ratings_seqs = df_ratings.groupby("User_id")["Title"].apply(lambda x: list(set(x.tolist()))).loc[lambda x: x.str.len() > 4].reset_index()
ratings_seqs = ratings_seqs["Title"]

def clear_nan(sample):
    return sample[np.array([all(isinstance(j, str) for j in seq) for seq in sample])]
ratings_seqs = clear_nan(ratings_seqs)
train = ratings_seqs.iloc[:30000]
test = ratings_seqs.iloc[30000:]


book_embds_dict = {titles[i]: np.array(book_embds[i], dtype=np.float32) for i in range(len(titles))}









def get_batch(split):
    def encode(book_title):
        return torch.tensor(book_embds_dict[book_title], dtype=torch.float32)
    def encode_seq(seq):
        return torch.cat([encode(title).unsqueeze(0) for title in seq], dim=0)

    data = train.reset_index(drop=True) if split == 'train' else test.reset_index(drop=True)
    ix = torch.randint(0, len(data), (batch_size,))

    seq = []
    y = []

    for i in ix:
        i = int(i)
        s = encode_seq(data[i][:-1])
        seq.append(s)
        y.append(encode(data[i][-1]))

    
    lengths = torch.tensor([len(s) for s in seq], dtype=torch.long)
    padded_seq = pad_sequence(seq, batch_first=True)
    packed_seq = pack_padded_sequence(padded_seq, lengths, batch_first=True, enforce_sorted=False)

    x = packed_seq.to(device)
    y = torch.stack(y).to(device)

    return x, y



model = Rekomendatel().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for _ in range(len(train) // batch_size):
        inputs, targets = get_batch('train')

        inputs, targets = inputs.cuda() if torch.cuda.is_available() else inputs, targets.cuda() if torch.cuda.is_available() else targets
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train):.4f}')
    
    scheduler.step(running_loss / len(train))

torch.save(model.state_dict(), 'model.pth')
