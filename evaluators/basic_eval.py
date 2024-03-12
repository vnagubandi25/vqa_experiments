import sys

questions_file = sys.argv[1]
guess_file = sys.argv[2]
acc = 0





for qid in questions_file:
    true_answer = questions_file[qid]['answer'].lower()
    guess_answer = guess_file[qid]['answer'].lower()

    if type(true_answer) == list:
        for answer in true_answer:
            if answer in guess_answer:
                acc += 1
                break
    else:
        if true_answer in guess_answer:
            acc += 1

print("acc: ", acc / len(questions_file))

