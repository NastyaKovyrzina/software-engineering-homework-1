import math
import numpy as np

from src.data import load_movies
from src.models import (
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeRecommender,
    GenreRecommender,
)

K = 10
MAX_USERS = 100  


def ranking_metrics(recommended, relevant):
    hits = [1 if movie_id in relevant else 0 for movie_id in recommended[:K]]
    precision = sum(hits) / K
    recall = sum(hits) / len(relevant)

    dcg = sum(hit / math.log2(position + 2) for position, hit in enumerate(hits))
    ideal = sum(1 / math.log2(position + 2) for position in range(min(K, len(relevant))))
    ndcg = dcg / ideal
    return precision, recall, ndcg


def main():
    movies, ratings, tags = load_movies()

    ratings = ratings.sort_values(["userId", "timestamp"])
    position = ratings.groupby("userId").cumcount()
    count = ratings.groupby("userId")["movieId"].transform("size")
    train = ratings[position < count * 0.8].copy()
    test = ratings[position >= count * 0.8].copy()

    known_movies = set(train["movieId"])
    known_users = set(train["userId"])
    test = test[test["movieId"].isin(known_movies) & test["userId"].isin(known_users)]
    movies = movies[movies["movieId"].isin(known_movies)].copy()
    last_train_time = train.groupby("userId")["timestamp"].max()
    tags = tags[tags["timestamp"] <= tags["userId"].map(last_train_time).fillna(-1)].copy()

    pop = PopularityRecommender().fit(train, movies)
    genre = GenreRecommender(pop).fit(movies, train)
    content = ContentBasedRecommender().fit(movies, tags)
    collab = CollaborativeRecommender(n_components=50).fit(train, movies)

    positives = test[test["rating"] >= 4].groupby("userId")["movieId"].apply(set).to_dict()
    histories = {user_id: group.sort_values("timestamp") for user_id, group in train.groupby("userId")}
    users = [user_id for user_id in positives
             if user_id in histories and (histories[user_id]["rating"] >= 4).any()]
    users = np.random.default_rng(42).choice(users, size=min(MAX_USERS, len(users)), replace=False)
    methods = ["Popularity", "Genre", "Content", "Collaborative"]
    scores = {name: [] for name in methods}
    shown = {name: set() for name in methods}
    pop_ids = pop.scores["movieId"].tolist()

    for user_id in users:
        relevant = positives[user_id]
        history = histories[user_id]
        liked = history[history["rating"] >= 4]

        seen = set(history["movieId"])
        seed_movie = int(liked.iloc[-1]["movieId"])
        recommendations = {
            "Popularity": [mid for mid in pop_ids if mid not in seen][:K],
            "Genre": [mid for mid in genre.recommend(movie_id=seed_movie, k=len(movies)) if mid not in seen][:K],
            "Content": [mid for mid in content.recommend(movie_id=seed_movie, k=len(movies)) if mid not in seen][:K],
            "Collaborative": collab.recommend(user_id=user_id, k=K),
        }

        for name, ids in recommendations.items():
            scores[name].append(ranking_metrics(ids, relevant))
            shown[name].update(ids)

    print(f"Обучение: {len(train)} оценок; проверка: {len(test)} оценок")
    print(f"Зрителей в проверке: {len(scores['Popularity'])}")
    print(f"Каталог для проверки: {len(movies)} фильмов")
    print(f"{'Метод':<16} {'Precision@10':>12} {'Recall@10':>11} {'NDCG@10':>10} {'Coverage':>10}")
    for name in methods:
        mean = np.mean(scores[name], axis=0)
        coverage = len(shown[name]) / len(movies)
        print(f"{name:<16} {mean[0]:>12.4f} {mean[1]:>11.4f} {mean[2]:>10.4f} {coverage:>10.2%}")

    users = test["userId"].map(collab.user_to_idx).to_numpy()
    items = test["movieId"].map(collab.movie_to_idx).to_numpy()
    actual = test["rating"].to_numpy()
    collab_predictions = np.sum(collab.user_factors[users] * collab.item_factors[items], axis=1)
    collab_predictions = np.clip(collab_predictions, 0.5, 5.0)
    pop_predictions = test["movieId"].map(pop.scores.set_index("movieId")["score"]).to_numpy()
    collab_rmse = np.sqrt(np.mean((actual - collab_predictions) ** 2))
    pop_rmse = np.sqrt(np.mean((actual - pop_predictions) ** 2))
    print(f"RMSE по всем {len(test)} тестовым оценкам: Popularity {pop_rmse:.4f}; Collaborative {collab_rmse:.4f}")


if __name__ == "__main__":
    main()
