from pre_train import embedding_matrix, vocab_size, embedding_dimension, padded_inputs, padded_outputs, int_to_tokens, tokens_to_int
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten,Input, BatchNormalization, LSTM, Attention, Embedding, Bidirectional, GRU, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split

'''
    after preprocessing my inputs and outputs are in the following format
    <start_sentence> some message <end_sentence>, this is then converted to a list of ints according to the tokens_to_int dictionary
'''
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
    super(Encoder, self).__init__()
    self.batch_size = batch_sz
    self.enc_units = enc_units
    self.embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], trainable = False)
    self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.lstm = LSTM(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.bidirectional_lstm = Bidirectional(self.lstm)

  def call(self, inputs, hidden_state, mode):
    '''if mode == 1 then program will use GRU, if mode == 2 then it will be vanilla lstm, if mode == 3 then it will use bidirectional_lstm'''
    x = self.embeddings(inputs)
    if mode == 1:
        output, state = self.gru(x, initial_state = hidden_state)
    if mode == 2:
        output, state_c, state_h = self.lstm(x) #uncomment this when using lstm, not passing initial_state right now
        state = [state_c, state_h] #uncomment when using lstm or bidirectional_lstm
    if mode == 3:
        output, forward_h, forward_c, backward_h, backward_c = self.bidirectional_lstm(x)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        state = [state_c, state_h] #uncomment when using lstm or bidirectional_lstm
    return output, state

  def get_initial_hidden_state(self):
    return tf.zeros((self.batch_size, self.enc_units))


maxlen_input = maxlen_output = padded_inputs.shape[1]
#Have a total of 6036 messages and each message is padded to a maxlen of 223, the input shape therefore is (223,)
print(padded_inputs.shape, padded_outputs.shape)

# Creating training and validation sets using an 80-20 split
x_train, x_val, y_train, y_val = train_test_split(padded_inputs, padded_outputs, test_size=0.2)

epochs = 200
buffer_size = len(x_train)
batch_size = 64
steps_per_epoch = buffer_size//batch_size
lstm_units = 1024 #embedding_dimension and vocab size are imported from pre_train
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

encoder = Encoder(vocab_size, embedding_dimension, lstm_units, batch_size, embedding_matrix)

example_input_batch, example_target_batch = next(iter(dataset))
# sample input
sample_hidden = encoder.get_initial_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden, 1)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
#print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


'''
have padded_inputs, padded_outputs
# TODO: 1)feed in the padded inputs into the embedding layers
        2)get the embedding representation, then feed in that representation in a bidirectional lstm
        3)feed in the outputs of the bidirectional lstm into the attention module to get context
        4)feed in the context into the post attention unidirectional lstm module to get the outputs
'''
