# E. Culurciello, Spetember 2022
# create a dataset from data extracted from PDF files (pdf_extractor.py)

# creates a caption.csv file similar to Flicker-8k and to be used with CLIP training

# ISSUES: 
# - find captions -- solved below with "figure N:" format search!
# - find end of page and where it resumes -- TBD

import os
import io
import csv
import glob
import shutil
import argparse
from nltk import tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

title = 'PDF document extractor - dataset generator'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--pdf_data_path', type=str, default="results/",  help='path for input PDF files')
    arg('--enable_tables', type=bool, default=False,  help='find tables?')
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

def create_dataset(args):
    isExist = os.path.exists(args.pdf_data_path)
    if not isExist:
        print("extracted pdf directory does not exist - run pdf_extractor.py first!")
        exit(1)

    isExist = os.path.exists(args.pdf_data_path+"/dataset")
    if not isExist:
        os.makedirs(args.pdf_data_path+"/dataset")
        os.makedirs(args.pdf_data_path+"/dataset/images/")

    # output csv file with [image, caption] pairs:
    CSV_HEADER = ["image","caption","id"]

    # output csv file with [image, caption] pairs:
    csv_file = open(args.pdf_data_path+'/captions.csv', 'w', encoding='UTF8')
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)

    figure_counter = 0 # counts figures for dataset

    # for every directory
    for directory in os.listdir(args.pdf_data_path):
        if directory.endswith(".pdf"):
            print(bcolors.HEADER + "Processing directory:", directory + bcolors.ENDC)

            # how many figures?
            figures = glob.glob1(args.pdf_data_path+directory+"/images/","*.png")
            for fig in figures:
                src = args.pdf_data_path+directory+"/images/"+fig
                dst = args.pdf_data_path+"/dataset/images/"+fig
                shutil.copyfile(src, dst)
                figure_counter+=1 # increment figure counter!

            figs_num = len(figures)
            print(bcolors.OKCYAN + "Number of figures: "+str(figs_num) + bcolors.ENDC)

            # open text:
            text_file = args.pdf_data_path+directory+"/text.txt"
            with open(text_file, encoding='utf8') as f:
                all_text = f.read()
                # print(all_text)
                sentences = tokenize.sent_tokenize(all_text)
                print(bcolors.OKCYAN + "Number of sentences: "+str(len(sentences)) + bcolors.ENDC)

                for fig in range(figs_num):
                    # find all sentence with references to the figures:
                    caption_counter = 0 # caption counter (id) for each figure
                    for s in sentences:
                        if s.lower().find("figure "+str(fig)) >= 0:
                            # check if this is the caption (has format: "figure N:") or similar:
                            if s.lower().find("figure "+str(fig)+":") >= 0 or \
                            s.lower().find("figure "+str(fig)+"-") >= 0:
                                # this is the caption~
                                print("\tFigure "+str(fig)+" CAPTION:" + bcolors.OKGREEN + s + bcolors.ENDC + str(caption_counter))
                            else:
                                print("\tFigure "+str(fig)+":" + bcolors.OKGREEN + s + bcolors.ENDC + str(caption_counter))

                            # csv data row:
                            row = [args.pdf_data_path+directory+"/figure_"+str(fig)+".png", s.replace('\n', ' '), str(caption_counter)]
                            writer.writerow(row)
                            caption_counter+=1 # increment caption id counter

    csv_file.close()


if __name__ == "__main__":
    args = get_args() # all input arguments
    print(title)
    create_dataset(args)