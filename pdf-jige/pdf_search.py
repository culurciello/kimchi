# E. Culurciello, May 2023
# search data extracted from PDF files (pdf_extractor.py)

import os
import argparse
import torch
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# sentence embedding:
from sentence_transformers import SentenceTransformer, util

title = 'PDF document extractor - dataset generator'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('i', type=str, default="multi-headed attention",  help='search text')
    arg('--pdf_data_path', type=str, default="results/",  help='path for input PDF files')
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


def search_pdf(args):
    isExist = os.path.exists(args.pdf_data_path)
    if not isExist:
        print("extracted pdf directory does not exist - run pdf_extractor.py first!")
        exit(1)

    print('You are searching for string:', args.i)

    # sentence embed model
    print('Loading SentenceTransformer...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('Done!')

    # embed input search string
    embed_input = model.encode(args.i, convert_to_tensor=True) # 1024 vector per sentence
    # print('embed_input.shape', embed_input.shape)

    # for every directory
    for directory in os.listdir(args.pdf_data_path):
        if directory.endswith(".pdf"):
            print(bcolors.HEADER + "Processing directory:", directory + bcolors.ENDC)

            # open text:
            text_file = args.pdf_data_path+directory+"/text.txt"
            embs_file = args.pdf_data_path+directory+"/embeddings.pth"

            isExist = os.path.exists(embs_file)
            if not isExist:
                print("embedding file does not exist - run pdf_embed.py first!")
                exit(1)

            # open embedding file:
            embeddings = torch.load(embs_file)

            with open(text_file, encoding='utf8') as f:
                all_text = f.read()
                # print(all_text)
                sentences = tokenize.sent_tokenize(all_text)
                print(bcolors.OKCYAN + "Number of sentences: "+str(len(sentences)) + bcolors.ENDC)

                # for all sentences:
                top_k=5
                cos_scores = util.cos_sim(embed_input, embeddings)[0]
                top_results = torch.topk(cos_scores, k=top_k)
                if args.print:
                    print("\nTop", top_k, "most similar sentences in corpus:")
                    for score, idx in zip(top_results[0], top_results[1]):
                        print("(Score: {:.4f})".format(score), sentences[idx])

                # print(sentences[top_results[1][0]], top_results[0][0])
        
        return sentences[top_results[1][0]], top_results[0][0]

if __name__ == "__main__":
    args = get_args() # all input arguments
    print(title)
    args.print=True # print here!
    matches, scores = search_pdf(args)