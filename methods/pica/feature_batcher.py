import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip
import torch
import torch.nn.functional as F
import os
import gc
import requests
from io import BytesIO
import sys

class CustomDataset(Dataset):
    def __init__(self, json_path):
        self.data = json.load(open(json_path, "r"))
        self.image_paths = []
        self.questions = []
        self.question_ids = []

        count = 0
        for id in self.data:
            query = self.data[id]
            self.question_ids.append(id)
            self.questions.append(query["question"]) 
            self.image_paths.append(query["image_filepath"])
            count = count + 1
        
        self.transform = clip.load("ViT-B/32", device="cuda")[1]

    def __len__(self):
        return len(self.image_paths)
    
    def load_image(self, image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    
    def __getitem__(self, idx):
        while True:
            try:
                image = self.load_image(self.image_paths[idx])
                image_input = self.transform(image)
                text_input = clip.tokenize([self.questions[idx]], truncate=True)
                question = self.question_ids[idx]
                text_input = text_input.squeeze(0)
                return question, image_input, text_input
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                print(question)
                idx += 1

                if idx >= len(self):
                    raise StopIteration("No more items in the dataset.")


dataset =  sys.argv[1]
output = sys.argv[2]

batch_size = 300
output_directory = output

dataset = CustomDataset(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

count = 1
for batch in dataloader:
    clip_model = clip.load('ViT-B/32', "cuda")[0]
    question_ids, image_inputs, text_inputs = batch

    image_inputs = image_inputs.to("cuda")
    text_inputs = text_inputs.to("cuda")

    image_features_batch = clip_model.encode_image(image_inputs)
    text_features_batch = clip_model.encode_text(text_inputs)

    data_to_save = {'image_features': image_features_batch, 'question_ids': question_ids, 'text_features': text_features_batch}
    print(data_to_save.keys())
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Save to the specified file format
    file_path = os.path.join(output_directory, f"batch_{count}.pt")
    torch.save(data_to_save, file_path)
    
    print(f"Saved batch {count} to: {file_path}")

    del image_inputs, text_inputs, image_features_batch, text_features_batch, clip_model, data_to_save
    torch.cuda.empty_cache()
    gc.collect()

    count += 1