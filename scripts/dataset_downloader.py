import wget
import os
from pathlib import Path
import zipfile
import uuid
import json


current_path = Path.cwd()
parent_path = current_path.parent
datasets_path = parent_path / "datasets"

vqav2_path = datasets_path / "vqav2"
gqa_path = datasets_path / "gqa"
okvqa_path = datasets_path / "okvqa"


vqav2 = {
    "annotations": {
        "link": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        "zip_path" : str(vqav2_path) + "/annotations.zip",
        "unzip_path" : str(vqav2_path) + "/annotations",
        "raw_file_path" : str(vqav2_path) + "/annotations" + "/v2_mscoco_val2014_annotations.json",
    },
    "questions": {
        "link" : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "zip_path" : str(vqav2_path) + "/questions.zip",
        "unzip_path" : str(vqav2_path) + "/questions",
        "raw_file_path" : str(vqav2_path) + "/questions/v2_OpenEnded_mscoco_val2014_questions.json",
        "formatted_file_path": str(vqav2_path)  + "/vqav2_qa_formatted.json"
    },
    "images": {
        "link": "http://images.cocodataset.org/zips/val2014.zip",
        "zip_path" : str(vqav2_path) + "/images.zip",
        "unzip_path" : str(vqav2_path) + "/images",
        "image_base_path": str(vqav2_path) + "/images" + "/val2014/COCO_val2014_"
    },
        
}

for file in vqav2:
    if file == "images":
        continue
    wget.download(vqav2[file]["link"], out=str(vqav2[file]["zip_path"]))
    with zipfile.ZipFile(vqav2[file]["zip_path"], 'r') as zip_ref:
        zip_ref.extractall(vqav2[file]["unzip_path"])

vqav2_raw_questions = json.load(open(vqav2["questions"]['raw_file_path'],'r'))['questions']

vqav2_formatted_questions = {}

for question in vqav2_raw_questions:
    vqav2_formatted_questions[question["question_id"]] = question
    path_id = 12 - len(str(vqav2_formatted_questions[question["question_id"]]['image_id']))
    vqav2_formatted_questions[question["question_id"]]['image_filepath'] = vqav2["images"]['image_base_path'] +'0'*path_id + str(vqav2_formatted_questions[question["question_id"]]['image_id'])+ ".jpg"

vqav2_raw_annotations = json.load(open(vqav2["annotations"]['raw_file_path'],'r'))['annotations']

for annotation in vqav2_raw_annotations:
    answers = []
    for answer in annotation['answers']:
        if answer["answer_confidence"]=="yes":
            answers.append(answer["answer"])
    vqav2_formatted_questions[annotation["question_id"]]['answer'] = list(set(answers))

json.dump(vqav2_formatted_questions,open(vqav2['questions']['formatted_file_path'],'w'),indent=1)

print("finished vqav2")


gqa = {
    "questions": {
        "link" : "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
        "zip_path" : str(gqa_path) + "/questions.zip",
        "unzip_path" : str(gqa_path) + "/questions",
        "raw_file_path" : str(gqa_path) + "/questions/testdev_balanced_questions.json",
        "formatted_file_path": str(gqa_path)  + "/gqa_qa_formatted.json"
    },
    "images": {
        "link": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
        "zip_path" : str(gqa_path) + "/images.zip",
        "unzip_path" : str(gqa_path) + "/images",
        "image_base_path": str(gqa_path) + "/images/images"
    },
        
}

for file in gqa:
    wget.download(gqa[file]["link"], out=str(gqa[file]["zip_path"]))
    with zipfile.ZipFile(gqa[file]["zip_path"], 'r') as zip_ref:
        zip_ref.extractall(gqa[file]["unzip_path"])


gqa_questions = json.load(open(gqa["questions"]['raw_file_path'],'r'))

for qid in gqa_questions:
    gqa_questions[qid]["image_filepath"] = gqa['images']['image_base_path'] + gqa_questions[qid]["imageId"] + '.jpg'

print("finished gqa")



json.dump(gqa_questions,open(gqa['questions']['formatted_file_path'],'w'),indent=1)

okvqa = {

    "annotations": {
        "link": "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip",
        "zip_path" : str(okvqa_path) + "/annotations.zip",
        "unzip_path" : str(okvqa_path) + "/annotations",
        "raw_file_path" : str(okvqa_path) + "/annotations" + "/mscoco_val2014_annotations.json",
    },
    "questions": {
        "link" : "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip",
        "zip_path" : str(okvqa_path) + "/questions.zip",
        "unzip_path" : str(okvqa_path) + "/questions",
        "raw_file_path" : str(okvqa_path) + "/questions/OpenEnded_mscoco_val2014_questions.json",
        "formatted_file_path": str(okvqa_path)  + "/okvqa_qa_formatted.json"
    },
    "images": {
        "link": "http://images.cocodataset.org/zips/val2014.zip",
        "zip_path" : str(okvqa_path) + "/images.zip",
        "unzip_path" : str(okvqa_path) + "/images",
        "image_base_path": str(okvqa_path) + "/images" + "/val2014/COCO_val2014_"
    },
        
}

for file in okvqa:
    wget.download(okvqa[file]["link"], out=str(okvqa[file]["zip_path"]))
    with zipfile.ZipFile(okvqa[file]["zip_path"], 'r') as zip_ref:
        zip_ref.extractall(okvqa[file]["unzip_path"])

okvqa_raw_questions = json.load(open(okvqa["questions"]['raw_file_path'],'r'))['questions']

okvqa_formatted_questions = {}

for question in okvqa_raw_questions:
    okvqa_formatted_questions[question["question_id"]] = question
    path_id = 12 - len(str(okvqa_formatted_questions[question["question_id"]]['image_id']))
    okvqa_formatted_questions[question["question_id"]]['image_filepath'] = okvqa["images"]['image_base_path'] +'0'*path_id + str(okvqa_formatted_questions[question["question_id"]]['image_id'])+ ".jpg"


okvqa_raw_annotations = json.load(open(okvqa["annotations"]['raw_file_path'],'r'))['annotations']

for annotation in okvqa_raw_annotations:
    answers = []
    for answer in annotation['answers']:
        if answer["answer_confidence"]=="yes":
            answers.append(answer["answer"])
    okvqa_formatted_questions[annotation["question_id"]]['answer'] = list(set(answers))

json.dump(okvqa_formatted_questions,open(okvqa['questions']['formatted_file_path'],'w'),indent=1)


print("finished okvqa")