import os
import io
import zipfile
import requests
import pandas as pd

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_ml(data_dir: str = "data") -> None:
    target = os.path.join(data_dir, "ml-latest-small", "ratings.csv")
    if os.path.exists(target):
        print(f"[data] уже есть: {target}")
        return
    os.makedirs(data_dir, exist_ok=True)
    print("[data] скачиваю MovieLens ml-latest-small...")
    r = requests.get(ML_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(data_dir)
    print("[data] готово.")


def load_movies(data_dir: str = "data/ml-latest-small"):
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    tags = pd.read_csv(os.path.join(data_dir, "tags.csv"))
    return movies, ratings, tags