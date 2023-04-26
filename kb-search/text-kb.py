# E. Culurciello
# April 2023
# knowledge base from text

import math
import pickle
import argparse
import torch
# import wikipedia
from pyvis.network import Network
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# filterNER https://huggingface.co/dslim/bert-base-NER
from transformers import AutoModelForTokenClassification, pipeline 

title = '>>> Generate knowledge base from text <<<'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('--text_filename', type=str, default="cthulhu-full.txt",  help='text file to search')
    arg('--filter', action='store_true',  help='filter for named entities')
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


class KB():
    def __init__(self):
        self.entities = {}
        self.relations = []
        # filterNER:
        self.filter_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        self.filter_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
        self.filter_nlp = pipeline("ner", model=self.filter_model, tokenizer=self.filter_tokenizer)

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def merge_relations(self, r1):
        r2 = [r for r in self.relations
              if self.are_relations_equal(r1, r)][0]
        spans_to_add = [span for span in r1["meta"]["spans"]
                        if span not in r2["meta"]["spans"]]
        r2["meta"]["spans"] += spans_to_add

    def filterNER(self, candidate_entity):
        if args.filter:
            ner_results = self.filter_nlp(candidate_entity)
            if len(ner_results) > 0:
                entity_data = {
                    "title": candidate_entity,
                    "url": candidate_entity,
                    "summary": candidate_entity
                }
                return entity_data
            else:
                return None
        else:
            entity_data = {
                    "title": candidate_entity,
                    "url": candidate_entity,
                    "summary": candidate_entity
                }
            return entity_data

    def add_entity(self, e):
        self.entities[e["title"]] = {k:v for k,v in e.items() if k != "title"}

    def add_relation(self, r):
        # filter with Name Entity Recognition
        candidate_entities = [r["head"], r["tail"]]
        entities = [self.filterNER(ent) for ent in candidate_entities]

        # if one entity does not exist, stop
        if any(ent is None for ent in entities):
            return

        # manage new entities
        for e in entities:
            self.add_entity(e)

        # rename relation entities with their wikipedia titles
        r["head"] = entities[0]["title"]
        r["tail"] = entities[1]["title"]

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def print(self):
        print(bcolors.OKBLUE+ "Entities:"+ bcolors.ENDC)
        for e in self.entities.items():
            print(bcolors.OKBLUE+f"  {e}")
        print("Relations:"+ bcolors.ENDC)
        for r in self.relations:
            print(bcolors.OKBLUE+f"  {r}"+ bcolors.ENDC)


def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations


def from_text_to_kb(text, span_length=128, verbose=False):
    # tokenize whole text
    inputs = tokenizer([text], return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(bcolors.OKGREEN + f"Input has {num_tokens} tokens"+ bcolors.ENDC)
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(bcolors.OKGREEN  + f"Input has {num_spans} spans"+ bcolors.ENDC)
    overlap = math.ceil((num_spans * span_length - num_tokens) / 
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(bcolors.OKGREEN  + f"Span boundaries are {spans_boundaries}"+ bcolors.ENDC)

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    # create kb
    kb = KB()
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                "spans": [spans_boundaries[current_span_index]]
            }
            kb.add_relation(relation)
        i += 1

    return kb

def save_kb(kb, filename):
    with open(filename, "wb") as f:
        pickle.dump(kb, f)

def load_kb(filename):
    res = None
    with open(filename, "rb") as f:
        res = pickle.load(f)
    return res


def save_network_html(kb, filename="network.html"):
    # create network
    net = Network(directed=True, width="1024px", height="1024px", bgcolor="#eeeeee")
    # print(net)

    # nodes
    color_entity = "#00FF00"
    for e in kb.entities:
        net.add_node(e, shape="circle", color=color_entity)

    # edges
    for r in kb.relations:
        net.add_edge(r["head"], r["tail"],
                    title=r["type"], label=r["type"])
        
    # save network
    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')
    net.show(filename, notebook=False)


if __name__ == "__main__":
    print(bcolors.HEADER + title + bcolors.ENDC)
    args = get_args() # all input arguments
    if args.filter:
        print(bcolors.OKGREEN + "Filtering name entity ON!"+ bcolors.ENDC)

    # Load model and tokenizer
    print(bcolors.OKGREEN + "loading models..."+ bcolors.ENDC)
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    print(bcolors.OKCYAN + "Processing file:", args.text_filename + bcolors.ENDC)
    with open(args.text_filename) as file:
        text = file.read()

    kb = from_text_to_kb(text, verbose=True)
    # kb = load_kb("cthulhu-kb.p")
    # kb.print()
    if args.filter:
        ft = "-filtered"
    else:
        ft=""
    basename = args.text_filename.split(".")[0]
    save_kb(kb, basename+ft+".p")
    save_network_html(kb, filename=basename+ft+".html")