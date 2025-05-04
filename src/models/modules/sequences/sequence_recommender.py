import json
import os
import sys
from difflib import get_close_matches
import pandas as pd

class SequenceRecommender:
    def __init__(self, dataset_path, sequence_path):
        self.dataset_df = pd.read_csv(dataset_path)
        self.filtered_patterns = self.load_sequences(sequence_path)
        self.titles = self.dataset_df["Title"].dropna().to_list()

    def closest_title(self, title, size):
        return get_close_matches(title, self.titles, n=size, cutoff=0.6)
    
    """placeholder for a better function to return closest titles to input"""
    def get_titles(self, title):
        res = []
        for t in self.titles:
            if title.lower() in t.lower():
                res.append(t)
        return res
    
    @staticmethod
    def load_sequences(path):
        with open(path, 'r') as f:
            raw_patterns = json.load(f)
        return [
            (tuple(item["pattern"]), item["support"])
            for item in raw_patterns if len(item["pattern"]) > 1
        ]

    @staticmethod
    def remove_duplicates(book_titles):
        titles_to_keep = []
        for title in book_titles:
            if not any(title.lower() in other.lower() or other.lower() in title.lower()
                       for other in titles_to_keep):
                titles_to_keep.append(title)
        return titles_to_keep

    """top 10 books found in the most of other users"""
    def recommend_seq(self, given_book):
        sequences_with_book = [
            (pattern, support)
            for pattern, support in self.filtered_patterns
            if given_book.lower() in [book.lower() for book in pattern]
        ]

        if not sequences_with_book:
            print(f"No sequence contains the book: '{given_book}'")
            return []

        sorted_sequences = sorted(sequences_with_book, key=lambda x: x[1], reverse=True)
        all_titles = [
            title for pattern_titles, _ in sorted_sequences
            for title in pattern_titles if title.lower() != given_book.lower()
        ]
        return self.remove_duplicates(all_titles)[:10]
