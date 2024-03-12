import json
import torch
from collections import OrderedDict
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

# similarity_file = defaults[dataset]["similarity_file"]
similarity = torch.load(input_dir)




new_similarity = {}

for x in similarity:
    sim = similarity[x]
    del sim[x]
    sorted_examples = dict(sorted(sim.items(), key=lambda item: item[1]))
    new_similarity[x] = sorted_examples


torch.save(new_similarity,output_dir)

