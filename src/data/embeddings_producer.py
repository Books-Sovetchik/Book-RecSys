import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import os

class EmbeddingsProducer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def create_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
    
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
    
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            masked_embeddings = token_embeddings * mask
            summed = masked_embeddings.sum(dim=1)
            counts = mask.sum(dim=1)
            mean_pooled = summed / counts
    
        return [emb.cpu().numpy() for emb in mean_pooled]
    



def produce_embeddings_batched(data, embeddings_path,  batch_size=16):
    from tqdm import tqdm
    embeddings_producer = EmbeddingsProducer()

    data.loc[:, "description"] = data["description"].fillna("No description")
    descriptions = data["description"].tolist()
    titles = data["Title"].to_numpy()

    embeddings = []

    for i in tqdm(range(0, len(descriptions), batch_size), desc="Embedding..."):
        batch_texts = descriptions[i:i + batch_size]
        batch_embeddings = embeddings_producer.create_embeddings_batch(batch_texts)
        embeddings.extend(batch_embeddings)

    embeddings = np.stack(embeddings).astype(np.float32)
    np.savez(embeddings_path, titles=titles, embeddings=embeddings)
    print("Embeddings saved to embeddings_structured.npz")
    return embeddings


import pandas as pd

#switch to real paths
fs_path = "/kaggle/input/f-books/LEHABOOKS.csv"
ss_path = "/kaggle/input/f-books/books_data.csv"
fs_embeddings_path = os.path.join("/kaggle", "working", "fs_embds.npz")
ss_embeddings_path = os.path.join("/kaggle", "working", "ss_embds.npz")

fs_data = pd.read_csv(fs_path)
ss_data = pd.read_csv(ss_path)

produce_embeddings_batched(fs_data, fs_embeddings_path)
produce_embeddings_batched(ss_data, ss_embeddings_path)

print("FINISHED")