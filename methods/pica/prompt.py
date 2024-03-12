import json
import json
import torch
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image
import requests
from io import BytesIO
from data_config import dataset ,defaults
import random
import time
import sys

question_file = sys.argv[1]
captions_file = sys.argv[2]
similarity_file = sys.argv[3]
save_path = sys.argv[4]

questions = json.load(open(question_file,'r'))
similarity = torch.load(similarity_file)
captions = json.load(open(captions_file,'r'))

def create_question(question_id, n_examples):
    examples = similarity[question_id]
    count = 0
    top_n = []
    for key, value in reversed(examples.items()):
        top_n.append(key)
        count = count +1
        if count==n_examples:
            break
    
    top_questions = [questions[item]["question"] for item in top_n]
    top_captions = [captions[item]["answer"] for item in top_n]
    top_answers = [questions[item]["answer"] for item in top_n]

    question = 'Please answer the question according to the above context.\n===\n'

    for x in range(len(top_questions)):
        question = question + f"Caption: {top_captions[x]}\nQuestion: {top_questions[x]}\nAnswer: {top_answers[x]}\n"
    
    question = question + "Caption: " + captions[question_id]["answer"] + "\n" + "Question:" + captions[question_id]["question"] + "\n" + "Answer:"
    return question



pica_prompts = {}

# Randomly sample 200 questions
sampled_questions = random.sample(questions.keys(),200)


for x in sampled_questions:
    question = create_question(x,8)
    pica_prompts[x] = question

json.dump(pica_prompts,open(save_path,'w'),indent=3)
