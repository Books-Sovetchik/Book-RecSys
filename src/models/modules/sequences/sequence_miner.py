import re
import json
import numpy as np
from collections import Counter, defaultdict
from pymining import seqmining

class SequenceMiner:
    def __init__(self, df_ratings):
        self.df = df_ratings.copy()
        self.id_to_title = {}
        self.sequences = []
        self.mined_patterns = []

    @staticmethod
    def normalize_title(title):
        title = re.sub(r'[\(\[\{].*?[\)\]\}]', '', title)
        title = title.replace('-', ' ')
        title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        return title.strip().lower()

    def preprocess(self):
        self.df['Title'] = self.df['Title'].astype(str).str.strip()
        self.df['NormalizedTitle'] = self.df['Title'].apply(self.normalize_title)

        normalized_to_originals = defaultdict(list)
        for original, normalized in zip(self.df['Title'], self.df['NormalizedTitle']):
            normalized_to_originals[normalized].append(original)

        normalized_to_best_original = {
            norm: Counter(titles).most_common(1)[0][0]
            for norm, titles in normalized_to_originals.items()
        }

        unique_norm_titles, normalized_ids = np.unique(self.df['NormalizedTitle'], return_inverse=True)
        self.df['NormalizedID'] = normalized_ids

        self.id_to_title = {
            idx: normalized_to_best_original[title]
            for idx, title in enumerate(unique_norm_titles)
        }

    def build_sequences(self, min_len=6, common_threshold=0.95):
        user_sequences = defaultdict(list)
        for user_id, book_id in zip(self.df['User_id'].values, self.df['NormalizedID'].values):
            user_sequences[user_id].append(book_id)

        def deduplicate(seq):
            seen = set()
            return [x for x in seq if not (x in seen or seen.add(x))]

        ratings_seqs = [deduplicate(seq) for seq in user_sequences.values()]
        ratings_seqs = [seq for seq in ratings_seqs if len(seq) >= min_len]

        book_counter = Counter(book for seq in ratings_seqs for book in set(seq))
        total_seqs = len(ratings_seqs)
        too_common_books = {
            book for book, count in book_counter.items()
            if count / total_seqs > common_threshold
        }

        filtered_seqs = [
            [book for book in seq if book not in too_common_books]
            for seq in ratings_seqs
        ]
        self.sequences = [seq for seq in filtered_seqs if len(seq) >= min_len]

    def mine_frequent_sequences(self, min_support=10, support_threshold=3, batch_size=300):
        global_counter = defaultdict(int)

        for i in range(0, len(self.sequences), batch_size):
            batch = self.sequences[i:i + batch_size]
            batch_result = seqmining.freq_seq_enum(batch, min_support=min_support)
            for pattern, support in batch_result:
                global_counter[pattern] += support

        final_result = [
            (pattern, support) for pattern, support in global_counter.items()
            if support >= support_threshold
        ]
        final_result.sort(key=lambda x: x[1], reverse=True)

        self.mined_patterns = [
            {
                "pattern": [self.id_to_title[book_id] for book_id in pattern],
                "support": support
            }
            for pattern, support in final_result if len(pattern) > 1
        ]

    def save_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.mined_patterns, f, indent=2)
