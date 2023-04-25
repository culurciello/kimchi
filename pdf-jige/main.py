# E. Culurciello, April 2023
# demo pdf semantic search 

import argparse
import streamlit as st

from pdf_extractor import extract_from_pdf
from pdf_dataset import create_dataset
from pdf_embed import create_dataset_embeddings
from pdf_search import search_pdf

title = "🔍 Semantic document explorer"

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--i', type=str, default="multi-headed attention",  help='search text')
    arg('--pdf_data_path', type=str, default="results/",  help='path for input PDF files')
    args = parser.parse_args()
    return args

args = get_args()

# title
st.title(title)
intro = st.empty()
with intro.container():
    # Text body
    st.text("Load a document and make them searchable")
    st.info(
        'To get started, please provide the required information below.',
        icon="ℹ️")

# step 1: ask user to input the documentto search
# uploaded_file = st.file_uploader("Choose a file")
# print(uploaded_file)

# if uploaded_file is not None:
#     try:
    #     # step 2: parse document into text and embeddings
    # except UnicodeDecodeError:
    #         st.warning(
    #             """
    #             🚨 The file doesn't seem to load. Check the filetype, file format and Schema
    #             """
    #         )

# step 3: ask user to input the search query
search_string = st.text_area("Provide text to search")
if search_string is not "":
    try:
        args.i = search_string
        # step 4: search
        matches, scores = search_pdf(args)
    except UnicodeDecodeError:
            st.warning(
                """
                🚨 Please write a search string
                """
            )
    # step 5: display results
    if matches is not None:
        #  print(matches, scores)
        st.divider()
        display_state = st.header("Matches and scores:")
        display_matches_state = st.text(matches)
        display_scores_state = st.text(scores)