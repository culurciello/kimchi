# E. Culurciello
# April 2023
# ABRAIN web version neural network
# huggingface version

# create dataset
# we use two drectories train/ and test/ filled with a HTML web page 
# and its rendering as a PNG image. We create a dataset with (image, html=label)

import os
import json


def gen_data(mode='train'):
    webpages = []; htmls = []
    in_dir = "../data/web/train/" if mode == 'train' else "../data/web/dev/"

    for filename in os.listdir(in_dir):
        f = os.path.join(in_dir, filename)
        # checking if it is a file
        pre_f, ext_f = os.path.splitext(f)
        if os.path.isfile(f):
            if f.endswith('.png'):
                print(f'processing file: {f}')
                html_file = open(pre_f+".html", "r")
                html_data = html_file.read()
                webpages.append(f); htmls.append(html_data)
        
    dd = []
    for webpage, html in zip(webpages, htmls):
        nd = {}
        nd['file_name'] = os.path.basename(webpage)
        nd['label'] = html
        dd.append(nd)

    with open('../data/web/{}/metadata.jsonl'.format(mode), "w", encoding="utf-8") as f:
        for item in dd:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    gen_data('train')
    gen_data('dev')
    print('Dataset created!')
    