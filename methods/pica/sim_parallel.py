import json
import torch
import torch.nn.functional as F
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys

def calculate_cosine_similarity(feature1, feature2):

    feature1 = feature1.unsqueeze(0)
    feature2 = feature2.unsqueeze(0)

    # Normalize features before calculating cosine similarity
    feature1 = F.normalize(feature1, p=2, dim=-1)
    feature2 = F.normalize(feature2, p=2, dim=-1)
    # Calculate cosine similarity
    similarity = F.cosine_similarity(feature1, feature2)

    return similarity.item()


def calculate_single_similarity(question_id, features_dict):
    sim = {}
    current_text = features_dict[question_id]["text_feature"]
    current_image = features_dict[question_id]["image_feature"]

    for ques in features_dict:
        comp_text_feature = features_dict[ques]["text_feature"]
        comp_image_feature = features_dict[ques]["image_feature"]
        avg_sim = (calculate_cosine_similarity(current_text, comp_text_feature) + calculate_cosine_similarity(current_image, comp_image_feature)) / 2
        sim[ques] = avg_sim

    return question_id, sim


def calculate_similarity_parallel(features_dict):
    similarity_dict = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calculate_single_similarity, question_id, features_dict) for question_id in features_dict]

        for i, future in enumerate(futures):
            question_id, sim = future.result()
            print("Finished processing question:", i + 1, "/", len(futures), "ID:", question_id)
            similarity_dict[question_id] = sim

    return similarity_dict

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    multiprocessing.set_start_method('spawn', force=True)
    num_processes = multiprocessing.cpu_count()  # Get the number of CPU cores
    saved_batches_path = input_dir
    output_json_path_parallel = output_dir

    saved_batches = []

    for i in range(1, 18):
        file_path = f"{saved_batches_path}/batch_{i}.pt"
        saved_batch = torch.load(file_path)
        saved_batches.append(saved_batch)

    result_dict = {}

    for x in saved_batches:
        question_ids = x['question_ids']
        text_features = x['text_features']
        image_features = x['image_features']

        for y in range(len(question_ids)):
            # Detach the tensors to avoid serialization issues
            text_feature = text_features[y].detach()
            image_feature = image_features[y].detach()

            temp = {"text_feature": text_feature, "image_feature": image_feature}
            result_dict[question_ids[y]] = temp

    print("After all the batches have been loaded", torch.cuda.memory_allocated("cuda") / 1024**2, "MB------------------")
    print(len(result_dict.keys()))

    similarity_dict_parallel = calculate_similarity_parallel(result_dict)

    # json.dump(similarity_dict_parallel, open(output_json_path_parallel,'w'), indent =2)
    torch.save(similarity_dict_parallel,output_json_path_parallel)
    print("we have saved")