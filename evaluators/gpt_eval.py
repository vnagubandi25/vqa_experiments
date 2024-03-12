import sys
import openai
from openai import OpenAI


def chatgpt(question,true, guess):
    prompt = f""" I have a question and 2 answers named Answer1 and Answer2. 
    Answer1 is the true answer to the question.
    Answer2 is a guess at the true answer.
    Here are the question and answer pairs
    Question:{question}
    Answer1:{true}
    Answer2:{guess}
    Determine if Answer1 and Answer2 are similar in the way that they answer the question. 
    Specifically check if Answer2 is equivalent to Answer1. 
    Note that it doesn't matter if Answer2 has a lot more detail even if its unnecessary details. 
    Please answer with Yes if the answers are similar and No if they aren't. Also explain why they answers aren't similar.   
            """
    # prompt = "I have a question and 2 answers evaluate whether both the answers are similar. Here is the question and answer pairs Question:" + question+ "Answer1:" + answer1 + "Answer2:" + answer2 + "Please answer with Yes if the answers are similar and No if they aren't. Also explain why they answers aren't similar."
    client = OpenAI(
    # This is the default and can be omitted
        api_key = ''
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            }
        ],
        model="gpt-4-0125-preview",
    )

    chatgpt_response = chat_completion.choices[0].message
    answer = chatgpt_response.content
    if "yes" in answer.lower():
        return 1
    else:
        return 0


questions_file = sys.argv[1]
guess_file = sys.argv[2]
acc = 0

for qid in questions_file:
    question = questions_file[qid]['question']
    true_answer = questions_file[qid]['answer'].lower()
    guess_answer = guess_file[qid]['answer'].lower()

    if type(true_answer) == list:
        for answer in true_answer:
            if chatgpt(question, true_answer, answer) == 1:
                acc += 1
                break
    else:
        if chatgpt(question, true_answer, guess_answer) == 1:
            acc += 1

print("acc: ", acc / len(questions_file))