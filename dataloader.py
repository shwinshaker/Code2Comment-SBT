from torch.utils.data import Dataset, DataLoader
import utils
import numpy as np
import torch

class musicDataset(Dataset):
	def __init__(self, inputs, target, encoder):
		self.inputs = inputs
		self.target = target
		self.encoder = encoder


	def __len__(self):
		# Return the total number of data samples
		return len(self.inputs)

	def __getitem__(self, ind):
		"""Returns one-hot encoded version of the target and labels
		"""
		data = self.inputs[ind]
		label = self.target[ind]
		# convert to one hot encoding
		x = []
		for w in data:
			x.append(self.encoder.get_one_hot(w))
		y = []
		# for w in label:
		# 	y.append(self.encoder.get_one_hot(w))
		for w in label:
			y.append(self.encoder.character_to_index[w])
		# print(torch.tensor(x, dtype=torch.float).size())
		# print(torch.tensor(y, dtype=torch.float))
		return (torch.tensor(x, dtype=torch.float), 
				torch.tensor(y, dtype=torch.long))


# load input data
def load_input_label(filepath):
	with open(filepath, 'r') as fp:
		input = fp.read()
	# get the first character
	first_ch = input[0]
	target = input[1:]
	target += first_ch
	return input, target


# divide corpus into chunks
def toChunk(chunk_size, corpus):
	chunk = len(corpus) // chunk_size
	result = []
	for i in range(chunk):
		result.append(corpus[i * chunk_size:(i + 1) * chunk_size])
	return result


def createLoaders(chunk_size=100, batch_size=1, extras={}):
	encoder = utils.String_Encoder()
	# load training, validation and test text
	train_input, train_target = load_input_label('data/train.txt')
	val_input, val_target = load_input_label('data/valid.txt')
	test_input, test_target = load_input_label('data/test.txt')
	encoder.encode(train_input)
	train_input = toChunk(chunk_size, train_input)
	train_target = toChunk(chunk_size, train_target)
	val_input = toChunk(chunk_size, val_input)
	val_target = toChunk(chunk_size, val_target)
	test_input = toChunk(chunk_size, test_input)
	test_target = toChunk(chunk_size, test_target)

	# Convert into dataloader
	train_dataset = musicDataset(train_input, train_target, encoder)
	val_dataset = musicDataset(val_input, val_target, encoder)
	test_dataset = musicDataset(test_input, test_target, encoder)

	num_workers = 0
	pin_memory = False
	# If CUDA is available
	if extras:
		num_workers = extras["num_workers"]
		pin_memory = extras["pin_memory"]

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
	return (train_dataloader, val_dataloader, test_dataloader), encoder
