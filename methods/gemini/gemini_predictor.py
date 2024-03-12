import os
import PIL.Image
import google.generativeai as genai
import json
from ratelimit import limits, sleep_and_retry
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
from io import BytesIO
from pathlib import Path
import os
import sys




# Define the rate limit: 60 calls per 70 seconds
CALLS = 60
RATE_LIMIT = 70
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)


@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def check_limit():
    '''Empty function just to check for calls to API'''
    pass
def encode_image(image_path):
    if image_path.startswith("http") or image_path.startswith("https"):
        response = requests.get(image_path)
        return PIL.Image.open(BytesIO(response.content))
    return PIL.Image.open(image_path)

def gemini_call(prompt,image_path):
        check_limit()
        model = genai.GenerativeModel('gemini-pro-vision')
        try:
            image = encode_image(image_path)
        except:
             return "image_loading_error"
        try:
            response1 = model.generate_content([prompt, image ],
                                                safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
    )
            answer1 = response1.text
            return answer1
        except:
             return "content_generation_error"
        


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
        answer = gemini_call(question,image_filepath)
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
            
     

      
          
     

