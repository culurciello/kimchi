# E. Culurciello, Spetember 2022
# extract tables, images, text from PDF file
# inspired from: https://www.thepythoncode.com/article/extract-pdf-images-in-python

import os
import io
import argparse
from PIL import Image

import fitz # PyMuPDF: https://pymupdf.readthedocs.io/en/latest/
# import camelot # find tables: https://github.com/camelot-dev/camelot


title = 'PDF document extractor - dataset generator'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--pdfs_path', type=str, default="test_pdfs/",  help='path for input PDF files')
    arg('--results_path', type=str, default="results/",  help='results path')
    arg('--enable_tables', type=bool, default=False,  help='find tables?')
    arg('--save', type=bool, default=True,  help='save')
    args = parser.parse_args()
    return args

def extract_from_pdf(args):
    if args.save:
        isExist = os.path.exists(args.results_path)
        if not isExist:
            os.makedirs(args.results_path)

    for file in os.listdir(args.pdfs_path):
        if file.endswith(".pdf"):

            file_name = os.path.basename(file)
            print(file_name)

            # open the file
            print('Opening PDF:', file)
            pdf_file = fitz.open(args.pdfs_path + file)
            print('Number of pages:', len(pdf_file))

            # this file result path:
            file_results_path = args.results_path + file_name + "/"

            if args.save:
                isExist = os.path.exists(file_results_path)
                if not isExist:
                    os.makedirs(file_results_path)
                isExist = os.path.exists(file_results_path+"images/")
                if not isExist:
                    os.makedirs(file_results_path+"images/")

            # iterate over PDF pages
            text_string = ""
            text_pages_list = []
            figures_list = []
            tables_list = []
            for page_index in range(len(pdf_file)):
                print('Processing page:', page_index+1)
                # get the page itself
                page = pdf_file[page_index]
                # 1- find text:
                text = page.get_text("text")
                print('\tText lenght:', len(text))
                print('\t\t', text)
                # text_pages_list.append(text)
                text_string = text_string + text #+ " \n~~~\n"

                # 2- find tables:
                if args.enable_tables:
                    tables = camelot.read_pdf(file, pages=str(page_index))
                    if len(tables)>0:
                        print('\tTables #:', len(tables))
                        for table in tables:
                            # print(table)
                            tables_list.append(table)
                            #save table:
                            savepath = results_path+f"table_{len(tables_list)}.csv"
                            if args.save:
                                tables[0].to_csv(savepath)


                # 3- find images
                image_list = page.get_images()
                # printing number of images found in this page
                if image_list:
                    print('\tImages #:', len(image_list))
                    for image_index, img in enumerate(image_list, start=1):
                        # get the XREF of the image
                        xref = img[0]
                        # extract the image bytes
                        base_image = pdf_file.extract_image(xref)
                        image_bytes = base_image["image"]
                        # get the image extension
                        image_ext = base_image["ext"]

                        # save image:
                        savepath = file_results_path+f"images/figure_{len(figures_list)+1}.{image_ext}"
                        figures_list.append(savepath)
                        if args.save:
                            image = Image.open(io.BytesIO(image_bytes)) # load it to PIL
                            # image.save(open(results_path+f"image{page_index+1}_{image_index}.{image_ext}", "wb"))
                            image.save(open(savepath, "wb"))


            # save extracted text:
            if args.save:
                with open(file_results_path+"text.txt", "w") as fp:
                    # pickle.dump(text_pages_list, fp)
                    fp.write(text_string)


if __name__ == "__main__":
    args = get_args() # all input arguments
    print(title)
    # print("NOTE: table extraction with camelot is not working well! Disabled by default!")
    extract_from_pdf(args)