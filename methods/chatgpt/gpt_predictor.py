from openai import OpenAI
import base64
import requests
import json
import sys
from pathlib import Path
import os


# # OpenAI API Key
api_key = ""


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_call(prompt,image_path):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    if "http" in image_path:
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        answer = response["choices"][0]['message']['content']
        return answer
    else:
        base64_image = encode_image(image_path)
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        answer = response["choices"][0]['message']['content']
        return answer


def generate(input_file,output_file):
    input = json.load(open(input_file,'r'))
    output = output_file
    questions = json.load(open(input,"r"))
    save_at = {}
    save_number = 50
    total_number = len(questions)

    for x in questions:
        question = questions[x]['question']
        image_filepath = questions[x]["image_filepath"]
        answer = gpt_call(question,image_filepath)
        save_at[x] =  {
             "answer": answer
        }
        save_number -= - 1
        total_number -= 1
        if save_at == 0:
            json.dump(save_at,open(output,'w'),indent=1)
            if total_number >= 50:
                save_at = 50
            else:
                save_at = total_number

    json.dump(save_at,open(output,'w'),indent=1)

def test(f,u):
    print(f,u)

