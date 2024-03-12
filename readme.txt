Setup your environemnt using the following commands:
1. conda create -n vqa_experiments python=3.10 -y 
2. python -m pip install -r requirements.txt


There are 4 methods that are implemented in LLaVa1.6, Gemini, ChatGPT, PICA 
along with 4 datasets AGVQA, OKVQA, GQA, VQAv2

The AGVQA datasets comes with this directory and you can download and format the other datasets by running 
the datasets_downloader from the scripts directory.

You can run the different methods and datasets by running the main.py file. It requires 3 arguments: method, datasets and output_filepath. 
PICA method has a different process described later.The documentation for which is in the pica methods file.

You can evaluate the datasets in two different ways basic_eval which will do an exact match search in the answer and 
gpt_eval which will use chatgpt to check whether the ground truth answer is equivalent to the answer predicted by the different methods.
both of these files will take the answers file and qa_answers file.




