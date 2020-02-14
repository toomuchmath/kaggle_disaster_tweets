import numpy as np
import os
import maplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import preprocessing


sentences = ["I love my dog.",
             "I love my penguin."]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print(word_index)

sentences_seq = tokenizer.texts_to_sequences(sentences)
print(sentences_seq)