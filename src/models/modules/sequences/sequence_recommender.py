import json
import os
import sys
from difflib import get_close_matches
import pandas as pd
import re

class SequenceRecommender:
    def __init__(self, dataset_path, sequence_path):
        self.dataset_df = pd.read_csv(dataset_path)
        self.filtered_patterns = self.load_sequences(sequence_path)
        self.titles = self.dataset_df["Title"].dropna().to_list()
    
    def normalize_title(self, title):
        title = re.sub(r'[\(\[\{].*?[\)\]\}]', '', title)
        title = title.replace('-', ' ')
        title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        return title.strip().lower()
    
    def get_books(self):
        all_titles = [title for pattern, _ in self.filtered_patterns for title in pattern]
        return all_titles

    def find_closest_title(self, input_title):
        normalized_input = self.normalize_title(input_title)
        all_titles = self.get_books()
        normalized_to_original = {
            self.normalize_title(title): title for title in all_titles
        }

        closest_matches = get_close_matches(normalized_input, normalized_to_original.keys(), n=1, cutoff=0.9)

        if closest_matches:
            return normalized_to_original[closest_matches[0]]
        else:
            return None

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
        given_book = self.find_closest_title(given_book)
        if (given_book is None):
            return None
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