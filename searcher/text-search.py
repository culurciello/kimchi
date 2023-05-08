# E. Culurciello, May 2023
# semantic search
# on text files

import os
import argparse
import torch
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# sentence embedding, q/a:
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

title = 'TEXT search'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--i', type=str, default="input.txt",  help='input text file')
    arg('--s', type=str, default="happy",  help='search string')
    args = parser.parse_args()
    return args

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def search_text_embeddings(args, debug=False):
    if debug:
        print('You are searching for string:', args.s)
    
    dir_name = os.path.dirname(args.i)
    file_name = os.path.basename(args.i)
    basename = file_name.split(".")[0]
    embedding_file_path = basename+".pth"
    dir_file_embs = dir_name + "/" + embedding_file_path
    isExist = os.path.exists(dir_file_embs)
    if not isExist:
        print("Embedding file does not exist - run text-embed.py first!")
        exit(1)

    # load embeddings:
    sentences, embeddings = torch.load(dir_file_embs)
    if debug:
        print('loaded:', dir_file_embs, "with embeddings shape:", embeddings.shape)

    # sentence embed model
    # print('Loading SentenceTransformer...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # print('Done!')

    # embed input search string
    embed_input = model.encode(args.s, convert_to_tensor=True)

    # search:
    top_k=5
    cos_scores = util.cos_sim(embed_input, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    if debug:
        print("\nTop", top_k, "most similar sentences in corpus:")
        for score, idx in zip(top_results[0], top_results[1]):
            print("(Score: {:.4f})".format(score), sentences[idx])
        
    return sentences[top_results[1][0]], top_results[0][0]

if __name__ == "__main__":
    args = get_args() # all input arguments
    print(title)
    matches, scores = search_text_embeddings(args, debug=True)

    # question / answer on search + text:
    print('you searched for:', args.s)
    question = args.s
    print('your match is:', matches)
    context = matches
    question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
    result = question_answerer(question=question, context=context)
    print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")