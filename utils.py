import argparse

def parse_arguments():

	parser = argparse.ArgumentParser()
	parser.add_argument('-wf', '--weights_file', metavar='FILE', type=str, 
		nargs=1, help='weights file for a pre-trained model')

	parser.add_argument('-ebt', '--embedding_type', type=str, 
		help='embedding type e.g., glove, bow')

	parser.add_argument('-ebd', '--embedding_dim', type=int, 
		help='embedding dimension')

	return parser.parse_args()

def log_to_file(msg):
    f = open('output.txt','a+')
    f.write(msg+'\n')
    f.close()