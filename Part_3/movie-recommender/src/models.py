import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Baseline: Popularity (weighted rating) ----------
class PopularityRecommender:
    """
    Эвристика: IMDb-style weighted rating.
    score = v/(v+m)*R + m/(v+m)*C
      R — средний рейтинг фильма, v — число оценок,
      C — средний рейтинг по всем фильмам, m — порог (80-й перцентиль).
    """

    def __init__(self, m_percentile: float = 0.8):
        self.m_percentile = m_percentile
        self.scores = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        stats = (
            ratings.groupby("movieId")["rating"]
            .agg(["count", "mean"])
            .reset_index()
        )
        C = stats["mean"].mean()
        m = stats["count"].quantile(self.m_percentile)
        stats["score"] = (
            stats["count"] / (stats["count"] + m) * stats["mean"]
            + m / (stats["count"] + m) * C
        )
        self.scores = stats.sort_values("score", ascending=False).reset_index(drop=True)
        return self

    def recommend(self, movie_id=None, user_id=None, k: int = 10):
        return self.scores.head(k)["movieId"].tolist()


# ---------- Content-based: TF-IDF на жанрах + тайтле + тегах ----------
class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf = None
        self.movies = None
        self.movie_to_idx = None

    def fit(self, movies: pd.DataFrame, tags: pd.DataFrame | None = None):
        self.movies = movies.reset_index(drop=True)
        self.movie_to_idx = {m: i for i, m in enumerate(self.movies["movieId"])}

        text = (
            self.movies["genres"].str.replace("|", " ", regex=False)
            + " "
            + self.movies["title"].fillna("")
        )

        if tags is not None and not tags.empty:
            tag_agg = (
                tags.groupby("movieId")["tag"]
                .apply(lambda s: " ".join(s.astype(str)))
                .reset_index()
                .rename(columns={"tag": "tag_text"})
            )
            merged = self.movies[["movieId"]].merge(tag_agg, on="movieId", how="left")
            text = text + " " + merged["tag_text"].fillna("")

        self.tfidf = self.vectorizer.fit_transform(text.fillna(""))
        return self

    def similar_movies(self, movie_id: int, k: int = 10):
        if movie_id not in self.movie_to_idx:
            return []
        idx = self.movie_to_idx[movie_id]
        sims = cosine_similarity(self.tfidf[idx], self.tfidf).flatten()
        order = np.argsort(-sims)
        order = [i for i in order if i != idx][:k]
        return self.movies.iloc[order]["movieId"].tolist()

    def recommend(self, movie_id=None, user_id=None, k: int = 10):
        if movie_id is None:
            return []
        return self.similar_movies(movie_id, k)


# ---------- Collaborative: item-item + user recs через SVD ----------
class CollaborativeRecommender:
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_factors = None
        self.item_factors = None
        self.item_factors_norm = None
        self.movies = None
        self.movie_to_idx = None
        self.user_ids = None
        self.user_to_idx = None
        self.X = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True)
        self.movie_to_idx = {m: i for i, m in enumerate(self.movies["movieId"])}
        self.user_ids = ratings["userId"].unique()
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}

        rows = ratings["userId"].map(self.user_to_idx).values
        cols = ratings["movieId"].map(self.movie_to_idx).values
        vals = ratings["rating"].values.astype(float)

        self.X = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.user_ids), len(self.movies)),
        )

        self.user_factors = self.svd.fit_transform(self.X)
        self.item_factors = self.svd.components_.T
        self.item_factors_norm = normalize(self.item_factors)
        return self

    def similar_movies(self, movie_id: int, k: int = 10):
        if movie_id not in self.movie_to_idx:
            return []
        idx = self.movie_to_idx[movie_id]
        sims = self.item_factors_norm @ self.item_factors_norm[idx]
        order = np.argsort(-sims)
        order = [i for i in order if i != idx][:k]
        return self.movies.iloc[order]["movieId"].tolist()

    def recommend_for_user(self, user_id: int, k: int = 10):
        if user_id not in self.user_to_idx:
            return []
        u = self.user_to_idx[user_id]
        scores = self.user_factors[u] @ self.item_factors.T
        rated = self.X[u].nonzero()[1]
        scores[rated] = -np.inf
        top = np.argsort(-scores)[:k]
        return self.movies.iloc[top]["movieId"].tolist()

    def recommend(self, movie_id=None, user_id=None, k: int = 10):
        if user_id is not None:
            return self.recommend_for_user(user_id, k)
        if movie_id is not None:
            return self.similar_movies(movie_id, k)
        return []


# ---------- Heuristic: top of the same genre ----------
class GenreRecommender:
    def __init__(self, popularity: PopularityRecommender):
        self.popularity = popularity
        self.movies = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies = movies.reset_index(drop=True).copy()
        pop = self.popularity.scores.set_index("movieId")["score"]
        self.movies["pop_score"] = self.movies["movieId"].map(pop).fillna(0.0)
        return self

    def recommend(self, movie_id=None, user_id=None, k: int = 10):
        if movie_id is None or movie_id not in self.movies["movieId"].values:
            return self.popularity.recommend(k=k)
        genres = (
            self.movies.loc[self.movies["movieId"] == movie_id, "genres"]
            .iloc[0]
            .split("|")
        )
        mask = self.movies["genres"].apply(
            lambda g: any(x in g.split("|") for x in genres)
        )
        pool = self.movies[mask & (self.movies["movieId"] != movie_id)]
        return pool.sort_values("pop_score", ascending=False).head(k)["movieId"].tolist()