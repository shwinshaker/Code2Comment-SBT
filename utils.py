'''
A data structure that efficiently get the index of a character and get the character given an index
in one-hot encoding
Stores a dictionary and a list
dictionary: character_to_index -> maps all characters in a corpus to a index in one-hot encoding
list: index_to_character -> returns the character of a given index in one-hot encoding
'''
import numpy as np

class String_Encoder:
	def __init__(self):
		self.character_to_index = {}
		self.index_to_character = []
		# self.counter = 0
		self.length = 0

	def encode(self, corpus):
		# for word in corpus:
		# 	if word not in self.character_to_index:
		# 		self.character_to_index[word] = self.counter
		# 		self.index_to_character.append(word)
		# 		self.counter += 1
		# self.length = self.counter

		self.character_to_index = dict([(c,i) for i,c in enumerate(set(corpus))])
		self.length = len(self.character_to_index)
		self.index_to_character = list(self.character_to_index.keys())
		assert(len(self.index_to_character) == self.length)

	# returns the one-hot encoding of a character
	def get_one_hot(self, word):
		one_hot = np.zeros(self.length)
		one_hot[self.character_to_index[word]] = 1
		return one_hot

	# return the character given an one-hot encoding index
	def get_character(self, index):
		return self.index_to_character[index]
