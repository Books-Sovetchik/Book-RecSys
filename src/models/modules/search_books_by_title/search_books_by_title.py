import pandas as pd
from difflib import get_close_matches


class SearchBooksByTitle:
    def __init__(self, library_path):
        self.dataset_df = pd.read_csv(library_path)
        self.titles = self.dataset_df["Title"].dropna().tolist()

    def closest_title(self, title, size=5):
        result = [tit for tit in self.titles if title.lower() in tit.lower()]

        similar = get_close_matches(title, self.titles, n=size, cutoff=0.5)

        combined = list(dict.fromkeys(result + similar))
        return combined