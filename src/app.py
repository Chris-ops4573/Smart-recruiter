from pipeline import rank_candidates
from helper.embed import embed

import streamlit as st
import tempfile
import sys
import os
import shutil

st.title("Smart recruiter")

uploaded = st.file_uploader(
    "Upload candidates",
    type=["json", "jsonl"]
)

num_candidates = st.number_input(
    "Number of candidates to rank",
    min_value=1,
    max_value=100,
    value=100
)

embeddings_dir = "embeddings_temp"
os.makedirs(embeddings_dir, exist_ok=True)

output_name = st.text_input(
    "Output file",
    "submission.csv"
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(uploaded.getvalue())
        path = f.name
    
        if st.button("Run"):
            with st.spinner("Generating embeddings..."):
                if(not os.path.exists("embeddings_temp/career.npy") or not os.path.exists("embeddings_temp/profile.npy")):
                    embed(path, embeddings_dir)

            with st.spinner("Ranking candidates..."):
                rank_candidates(num_candidates, path, output_name)

            with open(output_name, "rb") as f:
                st.download_button(
                    "Download Results",
                    data=f,
                    file_name=output_name,
                    mime="text/csv"
                )

                shutil.rmtree("embeddings_temp")