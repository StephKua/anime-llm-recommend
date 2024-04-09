import pandas as pd
import requests
from scipy.sparse import csr_matrix
import streamlit as st
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from sklearn.neighbors import NearestNeighbors

@st.cache_resource(show_spinner=False)
def load_data():
    # Read Data
    with st.spinner("Reading data..."):
        full_data = pd.read_parquet("./data/simple_anime_rating.gzip")
        
        # Filter Data for vote at least 200 and above
        valid_users = full_data.user_id.value_counts().reset_index()
        valid_users = valid_users.loc[valid_users['count']>=200].user_id.unique()
        full_data_valid_users = full_data[full_data.user_id.isin(valid_users)]

        # Convert data into item user matrix
        anime_pivot=full_data_valid_users.pivot_table(index='name',columns='user_id',values='user_rating').fillna(0)

        return anime_pivot


def get_wiki_data(title: str):

    data_path = Path("index_data")
    
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "prop": "extracts|search",
            "exlimit": "max",  # Retrieve extracts for all search results
            "explaintext": True,
            "generator": "search",
            "gsrsearch": title,
            "gsrlimit": 1,     # Limit the number of search results to 1
        },
    ).json()

    # Look thru search results
    try:
        pages = response["query"]["pages"]

        # Get the first page ID
        first_page_id = next(iter(pages))

        # Extract page content
        page_content = pages[first_page_id]["extract"]

        with open(data_path / f"{title}.txt", "w") as fp:
            fp.write(page_content)
    except Exception as e:
        with open(data_path / f"{title}.txt", "w") as fp:
            fp.write("")


def recommend(interested_title: str, anime_pivot: pd.DataFrame):
    with st.spinner(text="Getting Recommendation..."):

        non_similar_titles = anime_pivot.loc[~anime_pivot.index.str.lower().str.contains(interested_title.lower())].copy()
        anime_matrix = csr_matrix(non_similar_titles.values)

        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(anime_matrix)

        target_title = anime_pivot.loc[anime_pivot.index.str.lower().str.contains(interested_title.lower())].head(1).values
        if len(target_title) == 0:
            return []
        else:
            distances, indices = model_knn.kneighbors(target_title.reshape(1, -1), n_neighbors = 6)
        
        return indices.flatten()[1:]

@st.cache_resource(show_spinner=False)
def load_index():
    with st.spinner(text="Indexing data"):
        reader = SimpleDirectoryReader(input_dir="./index_data")
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs, show_progress=True)
        return index