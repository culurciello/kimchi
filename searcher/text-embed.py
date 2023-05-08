# E. Culurciello, May 2023
# semantic search
# on text files

import os
import argparse
import torch
from nltk import tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# sentence embedding:
from sentence_transformers import SentenceTransformer, util

title = 'TEXT data embedding'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--i', type=str, default="input.txt",  help='input text file')
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

def create_text_embeddings(args):
    # sentence embed model
    print('Loading SentenceTransformer...')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print('Done!')

    with open(args.i, encoding='utf8') as f:
        all_text = f.read()
        sentences = tokenize.sent_tokenize(all_text)
        print(bcolors.OKCYAN + "Number of sentences: "+str(len(sentences)) + bcolors.ENDC)

        print('Embedding all sentences...')
        embeddings = embedder.encode(sentences, convert_to_tensor=True)
        dir_name = os.path.dirname(args.i)
        file_name = os.path.basename(args.i)
        basename = file_name.split(".")[0]
        embedding_file_path = basename+".pth"
        dir_file_to_save = dir_name + "/" + embedding_file_path
        torch.save((sentences, embeddings), dir_file_to_save)
        print('Saved:', dir_file_to_save, "with shape:", embeddings.shape)


if __name__ == "__main__":
    args = get_args() # all input arguments
    print(title)
    create_text_embeddings(args)