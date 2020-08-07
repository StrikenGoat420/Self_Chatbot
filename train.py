from preprocessing_utils import open_file, fit_on_texts, texts_to_sequences, merge_inputs
from gensim.models import Word2Vec, KeyedVectors
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import Encoder, Decoder, Attention, train, loss_function
import time
import os

def get_inputs_and_outputs(messages, inputs, outputs):
    for m in messages:
        for i in range(len(m)):
            if i == 0:
                inputs.append(m[i])
            else :
                outputs.append(m[i])

def preprocess(sentence, token_to_int, pad_len):
  #sentence will be a list
  encoded_input = []
  for i in range(len(sentence)):
    sentence[i] = "<start_sentence> "+sentence[i]+" <end_sentence>"
  texts_to_sequences(sentence, tokens_to_int, encoded_input)
  padded_inputs = pad_sequences(encoded_input, maxlen=pad_len, padding='post')
  return padded_inputs


word_vectors = KeyedVectors.load(r'word2vec.kv', mmap='r')
messages = []
open_file(r'cleaned_data.csv', messages)

merged_messages = []
merge_inputs(messages, merged_messages, 5)#use of this function explained in preprocessing_utils.py
X = [] #input list
Y = [] #output list
get_inputs_and_outputs(merged_messages, X, Y)
X = list(np.array(X).flatten())

tokens_to_int = {}
fit_on_texts('word2vec.kv', tokens_to_int) #this function populates the token_to_int dict
'''forgot to end start sentence and end sentence token at the start and end of each sentence so doing that here'''
min_val = 0
for k,v in tokens_to_int.items():
    if v > min_val:
        min_val = v
#now we have the total number of words in the dict and just have to add two more tokens and assign the int min_val+1 and min_val+2 to them
tokens_to_int['<start_sentence>'] = min_val+1
tokens_to_int['<end_sentence>'] = min_val+2

int_to_tokens = {v : k for k,v in tokens_to_int.items()} #this is to retrive the word predicted by the nn later on

encoded_input = []
encoded_output = []
#right now the padding length has been selected as the max_len which is 221, which was known since before
#have to write solution which finds the max_len on its own
padded_inputs = preprocess(X, tokens_to_int, 65)
padded_outputs = preprocess(Y, tokens_to_int, 65)
print(padded_inputs[20])
print(padded_outputs[20])

vocab_size = len(tokens_to_int) + 1
embedding_dimension = word_vectors['sup'].shape[0]
embedding_matrix = np.zeros((vocab_size, embedding_dimension))

#defining the embedding matrix
for word, int_val in tokens_to_int.items():
    try:
        embedded_vector = word_vectors[word]
        embedding_matrix[int_val] = embedded_vector
    except KeyError:
        if word == '<start_sentence>':
            embedded_vector = np.zeros(word_vectors['sup'].shape)
            embedding_matrix[int_val] = embedded_vector
        elif word == '<end_sentence>':
            embedded_vector = np.ones(word_vectors['sup'].shape)
            embedding_matrix[int_val] = embedded_vector

maxlen_input = maxlen_output = padded_inputs.shape[1]
#Have a total of 6036 messages and each message is padded to a maxlen of 223, the input shape therefore is (223,)

# Creating training and validation sets using an 80-20 split
x_train, x_val, y_train, y_val = train_test_split(padded_inputs, padded_outputs, test_size=0.2)

#defining the hyper params
EPOCHS = 200
buffer_size = len(x_train)
batch_size = 8
steps_per_epoch = buffer_size//batch_size
lstm_units = 256
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

optimizer = tf.keras.optimizers.Adam()
loss_function_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

encoder = Encoder(vocab_size, embedding_dimension, lstm_units, batch_size, embedding_matrix)
decoder = Decoder(vocab_size, embedding_dimension, embedding_matrix, lstm_units, batch_size)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
print(dataset, x_train.shape, x_val.shape)

for epoch in range(EPOCHS):
    start = time.time()
    encoder_hidden = encoder.get_initial_hidden_state()
    total_loss = 0
    for (batch, (inp, output)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train(inp, output, encoder_hidden, encoder, decoder, batch_size, optimizer, loss_function_object)
        total_loss += batch_loss
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
