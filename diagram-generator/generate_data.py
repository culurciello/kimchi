# generate a bunch of diagrams and some captions automatically

import os
import glob
import random
import json
import graphviz

dataset_path = 'dataset/'
dpExist = os.path.exists(dataset_path)
if not dpExist:
  os.makedirs(dataset_path)

nodes = ["A","B","C","D","E","F","G","H","I","J"]

num_data = 8

gdata = {} # store graph data here

for sample in range(num_data):

    # get 4 nodes at random
    dot = graphviz.Digraph(name='my diagram', engine='neato')
    n = {}

    # generate 4 nodes diagram:
    n = random.sample(nodes, 4)
    for i in range(4):
        dot.node(n[i],n[i])

    dot.edges([n[0]+n[1], n[1]+n[2], n[2]+n[3], n[3]+n[0]])
    label = n[0]+" connects to "+n[1]+", "+n[1]+" connects to "+n[2]+", "+n[2]+" connects to "+n[3]+", "+n[3]+" connects to "+n[0]+"."

    q = random.randint(0,3)
    question = "What does "+n[q]+" connects to?"
    next_q = q+1
    if q == 3: next_q = 0
    answer = n[next_q]

    # outputs:
    # print(n)
    # print(label)
    # print(question, answer)
    # print(dot.source)
    dot.render(outfile='dataset/'+str(sample)+'.png')#, view=True)
    gdata[sample] = [n, label, question, answer]

# print(gdata)

with open("dataset/gdata.json", "w") as fp:
    json.dump(gdata, fp)

files = glob.glob('dataset/*.gv')
for f in files:
    os.remove(f)