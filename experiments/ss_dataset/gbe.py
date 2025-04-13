import sys
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import torch
from torch import nn
from torch import functional as F

batch_size = 32
block_size = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'


DATA_PATH = os.path.join(PROJECT_ROOT, "data")
PATH_RATINGS = os.path.join(DATA_PATH, "raw_data", "kaggle_second_sem", "books_rating.csv")
PATH_BOOKS = os.path.join(DATA_PATH, "raw_data", "kaggle_second_sem", "books_data.csv")
PATH_EMBDS = os.path.join(DATA_PATH, "embeddings", "expanded_embds_ss.npy")

df_books = pd.read_csv(PATH_BOOKS)
df_ratings = pd.read_csv(PATH_RATINGS)
book_embds = np.load(PATH_EMBDS, allow_pickle=True)

# Book embds has shape like
#(Title, (count, 1), (review/score, 1) (author, 1), (categories, 1), (publisher, 1), (description_embd, 384))


ratings_seqs = df_ratings.groupby("User_id")["Title"].apply(lambda x: list(set(x.tolist()))).loc[lambda x: x.str.len() > 5].reset_index()
train = ratings_seqs["Title"].iloc[:30000]
test = ratings_seqs["Title"].iloc[30000:]


torch.randint(2, (3, 3))

# torch.randint(len(train[1636]) - block_size, (1,))
torch.randint(len(train[20551]) - block_size, (1,))


book_embds_dict = {row[0]: np.array(row[1:], dtype=np.float32) for row in book_embds}

def encode(book_title):
    return book_embds_dict[book_title]
def encode_seq(seq):
    return [book_embds_dict[title] for title in seq]

def clear_nan(sample):
    return sample[np.array([all(isinstance(j, str) for j in seq) for seq in sample])]

train = clear_nan(train)
test = clear_nan(test)

train.shape, test.shape

def get_batch(split):
    data = train.reset_index(drop=True) if split == 'train' else test.reset_index(drop=True)
    ix = torch.randint(len(train), (batch_size,))

    x = []
    y = []
    for i in ix:
        i = int(i)       
        j = torch.randint(len(data[i]) - block_size, (1,))
        x.append(encode_seq(data[i][j:j + block_size]))
        y.append(encode_seq(data[i][j + 1:j + block_size + 1]))

    x = torch.tensor(x)
    y = torch.tensor(y)
    x, y = x.to(device), y.to(device)
    return x, y


# Получим максимальные индексы для категориальных признаков
def get_max_index(col_idx):
    return int(max(
        embd[1 + col_idx]
        for embd in book_embds_dict.values()
    ))

count_size = get_max_index(0) + 100000
review_size = get_max_index(1) + 100000
author_size = get_max_index(2) + 1000000
cat_size = get_max_index(3) + 100000
publisher_size = get_max_index(4) + 100000


count_embd_size = max(train)

sm_embd = 20
desc_embd = 384
n_embd = sm_embd * 5 + desc_embd


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, 4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class Optimus(nn.Module):
    def __init__(self):
        super().__init__()
        self.count_embds = nn.Embedding(count_size, sm_embd, device=device)
        self.review_embds = nn.Embedding(review_size, sm_embd, device=device)
        self.author_embds = nn.Embedding(author_size, sm_embd, device=device)
        self.cat_embds = nn.Embedding(cat_size, sm_embd, device=device)
        self.publisher_embds = nn.Embedding(publisher_size, sm_embd, device=device)
        self.pos_embds = nn.Embedding(block_size, n_embd, device=device)
        self.blocks = nn.Sequential(*[Block() for _ in range(4)])
        self.ln_head = nn.Linear(n_embd, 5 + desc_embd, device=device)
    def forward(self, x):
        B, T, C = x.shape
        x = torch.cat([
            self.count_embds(x[:, :, 0].long()),
            self.review_embds(x[:, :, 1].long()),
            self.author_embds(x[:, :, 2].long()),
            self.cat_embds(x[:, :, 3].long()),
            self.publisher_embds(x[:, :, 4].long()),
            x[:, :, 5:]
        ], dim=-1)

        pos_embd = self.pos_embds(torch.arange(T, device=x.device)).unsqueeze(0)  # shape: (1, T, n_embd)
        x = x + pos_embd

        x = self.blocks(x)
        x = self.ln_head(x)
        return x
    
model = Optimus().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

loss_fn = nn.MSELoss()

for step in range(10000):
    xb, _ = get_batch('train')  
    
    out = model(xb)

   
    x_target = out[:, 1:, :]    
    x_input = out[:, :-1, :]        

    loss = loss_fn(x_input, x_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")
