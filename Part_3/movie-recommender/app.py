import os
import pickle

import pandas as pd
import streamlit as st

ART_PATH = "artifacts/models.pkl"

st.set_page_config(page_title="Movie RecSys", layout="wide")
st.title("Movie Recommender System")
st.caption("Content-based · Collaborative (SVD) · Popularity · Genre ")


@st.cache_resource
def load_artifacts():
    with open(ART_PATH, "rb") as f:
        return pickle.load(f)


if not os.path.exists(ART_PATH):
    st.error("Сначала выполните `python train.py`")
    st.stop()

art = load_artifacts()
movies: pd.DataFrame = art["movies"]
recs = art["recommenders"]

st.sidebar.header("Настройки")
algo = st.sidebar.selectbox(
    "Алгоритм",
    ["Popularity", "Genre (heuristic)", "Content-based", "Collaborative (SVD)"],
)
k = st.sidebar.slider("Сколько рекомендаций (K)", 5, 30, 10)

tab_movie, tab_user, tab_pop = st.tabs(["По фильму", "По пользователю", "Популярное"])


def render(ids, header):
    st.subheader(header)
    if not ids:
        st.warning("Пусто")
        return
    res = movies[movies["movieId"].isin(ids)][["title", "genres"]].reset_index(drop=True)
    res["_o"] = res["title"].map(
        {movies.loc[movies["movieId"] == mid, "title"].iloc[0]: i for i, mid in enumerate(ids)}
    )
    res = res.sort_values("_o").drop(columns="_o")
    st.dataframe(res, use_container_width=True)


with tab_movie:
    titles = (
        movies[["movieId", "title"]]
        .drop_duplicates("title")
        .sort_values("title")
        .reset_index(drop=True)
    )
    selected_title = st.selectbox("Выберите фильм", titles["title"].tolist())
    movie_id = int(titles.loc[titles["title"] == selected_title, "movieId"].iloc[0])

    if st.button("Рекомендовать", key="btn_movie"):
        if algo == "Popularity":
            ids = recs["popularity"].recommend(k=k)
        elif algo == "Genre (heuristic)":
            ids = recs["genre"].recommend(movie_id=movie_id, k=k)
        elif algo == "Content-based":
            ids = recs["content"].recommend(movie_id=movie_id, k=k)
        else:  # Collaborative (SVD)
            ids = recs["collab"].recommend(movie_id=movie_id, k=k)
        render(ids, f"Top-{k} для «{selected_title}» ({algo})")

with tab_user:
    user_ids = sorted(recs["collab"].user_ids.tolist())
    user_id = st.selectbox("Выберите userId", user_ids)
    if st.button("Рекомендовать", key="btn_user"):
        ids = recs["collab"].recommend(user_id=int(user_id), k=k)
        render(ids, f"Top-{k} для пользователя {user_id} (Collaborative)")

with tab_pop:
    st.subheader(f"Топ-{k} популярных фильмов (weighted rating)")
    render(recs["popularity"].recommend(k=k), "Популярное")