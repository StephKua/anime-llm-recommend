__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from llama_index.core import Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from utils.util import load_data, get_wiki_data, load_index, recommend
import os
from tqdm import tqdm

st.header("What is your favorite anime? I will recommend 5 others for you.")
st.sidebar.title("How to use?")
st.sidebar.write("1. Enter favorite anime name, eg: One Piece")
st.sidebar.write("2. Wait for model recommendation and response")
st.sidebar.title("Reference")
st.sidebar.write("1. Embedding Model: Local BAAI/bge-small-en-v1.5")
st.sidebar.write("2. LLM Model: Local MISTRAL")
st.sidebar.write("3. Context Data from Wiki")
st.sidebar.write("4. MAL Recommendation Dataset")


# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# init models
if "model" not in st.session_state:
    st.session_state["model"] = ""

if "data" not in st.session_state:
    st.session_state['data'] = load_data()

# Set Local Embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# Set Local LLM
Settings.llm = Ollama(model="mistral", request_timeout=1800.0)

# Get Anime Wiki Data
temp = os.listdir("./index_data")
anime_names = st.session_state['data'].index.tolist()
if len(temp) == 0:
    with st.spinner("Getting Wiki Data, Estimate Time 7 mins"):
        for i in tqdm(anime_names, total=len(anime_names)):
            get_wiki_data(i)

# Load Index
index = load_index()
chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True, similarity_top_k=1)

anime_index = st.session_state['data']

# Prompt
if prompt := st.chat_input("Enter your favorite anime..."):
    # add latest message to history in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Perform Recommendation
    result = recommend(prompt, anime_index)
    with st.chat_message("assistant"):

        if len(result) == 0:
            message = st.write("Unable to retrieve recommendation. Please try any of following: 'One Piece', 'Naruto', 'Bleach'.")
        else:
            new_prompt = f"""
                I'm a good storyteller. 
                I always keep the interesting plot suspended and let user wonder what's so popular about the shows you recommending.

                Given your favorite anime is {prompt},

                I recommended {anime_index.index[result[0]]}, {anime_index.index[result[1]]}, {anime_index.index[result[2]]}, {anime_index.index[result[3]]} and {anime_index.index[result[4]]}
                First, start my response with the recommendation above.
                Next, convince you each of the anime above is similar with {prompt}.
                Finally, summarize the plot and the exciting part to you.
                Limit my response by 300 words.

                If there is no context explaining the show, please return "Go check it out, You might like it." 
            """
            response = chat_engine.stream_chat(new_prompt)

            message = st.write_stream(response.response_gen)
        st.session_state["messages"].append({"role": "assistant", "content": message})