import os
import pickle

from src.data import download_ml, load_movies
from src.models import (
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeRecommender,
    GenreRecommender,
)

ART_DIR = "artifacts"
ART_PATH = os.path.join(ART_DIR, "models.pkl")


def main():
    download_ml()
    movies, ratings, tags = load_movies()

    print("[train] popularity...")
    pop = PopularityRecommender().fit(ratings, movies)

    print("[train] genre heuristic...")
    genre = GenreRecommender(pop).fit(movies, ratings)

    print("[train] content-based...")
    content = ContentBasedRecommender().fit(movies, tags)

    print("[train] collaborative (SVD)...")
    collab = CollaborativeRecommender(n_components=50).fit(ratings, movies)

    os.makedirs(ART_DIR, exist_ok=True)
    with open(ART_PATH, "wb") as f:
        pickle.dump(
            {
                "movies": movies,
                "recommenders": {
                    "popularity": pop,
                    "genre": genre,
                    "content": content,
                    "collab": collab,
                },
            },
            f,
        )
    print(f"[train] сохранено: {ART_PATH}")


if __name__ == "__main__":
    main()