import os
import sys 

import numpy as np

ss_embeddings_path = os.path.abspath(os.path.join(os.getcwd(), "data", "embeddings", "expanded_embds_ss.npy"))
fs_embeddings_path = os.path.abspath(os.path.join(os.getcwd(), "data", "embeddings", "books_embeddings_dataset.npy"))

ss_dataset = np.load(ss_embeddings_path, allow_pickle=True)
fs_dataset = np.load(fs_embeddings_path, allow_pickle=True)

print(ss_dataset.shape)
print(fs_dataset.shape)