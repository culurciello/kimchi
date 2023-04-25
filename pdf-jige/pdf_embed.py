# E. Culurciello, November 2022
# embed text data extracted from PDF files (pdf_extractor.py)

import os
import argparse
import torch
from nltk import tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# sentence embedding:
from sentence_transformers import SentenceTransformer, util

title = 'PDF document extractor - data embedding'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
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

def create_dataset_embeddings(args):
    isExist = os.path.exists(args.pdf_data_path)
    if not isExist:
        print("extracted pdf directory does not exist - run pdf_extractor.py first!")
        exit(1)

    # sentence embed model
    print('Loading SentenceTransformer...')
    model = SentenceTransformer('stsb-roberta-large')
    print('Done!')

    # for every directory
    for directory in os.listdir(args.pdf_data_path):
        if directory.endswith(".pdf"):
            print(bcolors.HEADER + "Processing directory:", directory + bcolors.ENDC)

            # open text:
            text_file = args.pdf_data_path+directory+"/text.txt"
            with open(text_file, encoding='utf8') as f:
                all_text = f.read()
                # print(all_text)
                sentences = tokenize.sent_tokenize(all_text)
                print(bcolors.OKCYAN + "Number of sentences: "+str(len(sentences)) + bcolors.ENDC)

                print('Embedding all sentences...')
                embeddings = model.encode(sentences, convert_to_tensor=True) # 1024 vector per sentence
                # print('done!')
                embedding_file_path = args.pdf_data_path+directory+"/embeddings.pth"
                torch.save(embeddings, embedding_file_path)
                print('Saved:', embedding_file_path, "with shape:", embeddings.shape)



if __name__ == "__main__":
    args = get_args() # all input arguments
    print(title)
    create_dataset_embeddings(args)