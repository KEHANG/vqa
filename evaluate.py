import os

from data import DataLoaderDisk, get_embedding_matrix
from model import vqa_model
from utils import log_to_file, parse_arguments
from tqdm import tqdm

def evaluate(weights_file,
			embedding_type='glove',
			embedding_dim=300):

	# create data loader
	data_path = os.path.join(os.path.dirname(__file__), 'data')
	option = {
		'data_root': data_path,   # MODIFY PATH ACCORDINGLY
		'fine_size': 224,
		'word_embedding_length': 1024,
		'randomize': False
		}
	data_loader = DataLoaderDisk(**option)

	word_index = data_loader.tokenizer.word_index
	if embedding_type == 'glove':
		embedding_path = os.path.join(data_path, 'glove.6B', 
									  'glove.6B.{0}d.txt'.format(embedding_dim))
	embedding_matrix = get_embedding_matrix(word_index, embedding_type, embedding_path)

	seq_length = 25
	model_val = vqa_model(embedding_matrix, seq_length, dropout_rate=0.5, num_classes=3131)
	model_val.load_weights(weights_file)

	batch_size = 2000
	epochs = 1
	iters = int(data_loader.val_num*epochs/batch_size)
	val_accuracy = 0
	for iteration in tqdm(range(iters)):
		img_batch_val, que_batch_val, y_batch_val = data_loader.next_batch(batch_size, mode='val')
		val_score = model_val.test_on_batch([img_batch_val, que_batch_val], y_batch_val)
		val_accuracy += float(val_score[1])

		msg = 'iter = {0}, val acc: {1:03f}'.format(iteration, float(val_score[1]))
		log_to_file(msg)

	msg = 'Overall Accuracy on Validation-Set: {0}'.format(val_accuracy/iters)
	log_to_file(msg)

def main():

	args = parse_arguments()
	weights_file = args.weights_file[0]
	embedding_type = args.embedding_type
	embedding_dim = args.embedding_dim
	evaluate(weights_file, embedding_type, embedding_dim)

main()



