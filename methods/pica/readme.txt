feature_batcher.py: 
	required: an output directory, a json file with question fields and image_filepath
	returns: in the output directory specified you will get a set of files each containing the image and text features of each question

sim_paralel.py:
	required: directory of feature batches(outout of feature_batcher.py)
	returns: similarity_parallel.pt file which gives you the list of similar things to each question.

order_dict.py:
	required: similarity file

prompt.py
	required: qa_file, similarity_file, captions_file, output_file
	returns: a list of prompts that you can put into gpt 3.5