from pre_train import embedding_matrix, vocab_size, embedding_dimension, padded_inputs, padded_outputs, int_to_tokens, tokens_to_int
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten,Input, BatchNormalization, LSTM, Attention, Embedding, Bidirectional, GRU, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split
import time
import os
'''
# TODO: Individual components are working, but getting resourceexhaustederror when training. Have to find a solution for that
'''
'''
    after preprocessing, inputs and outputs are in the following format
    <start_sentence> some message <end_sentence>, this is then converted to a list of ints according to the tokens_to_int dictionary
    use https://www.tensorflow.org/tutorials/text/nmt_with_attention as reference
'''
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, num_units, batch_size, embedding_matrix):
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.enc_units = num_units
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

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        #we dont apply any activation to the dense layers, because we want the linearity, ie - we want (W1*hidden_state) and not tanh(W1*hidden_state)
        #we will apply the activation later when we have (W1*hidden_state) and (W2*encoder_output)
        #when we apply the activation to those two, we get an output of the shape (batch_size, max_length, units), but since we only want one score to simplify things
        #we use an additional linear layer called V, which only has one neuron, hence giving us an output of the shape (batch_size, max_length, 1)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, hidden_state, encoder_output):
        #shape of hidden_state = (batch_size, hidden_size) it is just the shape of the hidden_states we get from the encoder as the output
        #shape of expanded_hidden_state = (batch_size, 1, hidden_size) hidden_size is just the number of encoder units we have
        #shape of encoder_output = (batch_size, max_len, hidden_size)
        expanded_hidden_state = tf.expand_dims(hidden_state,1)
        #score shape == (batch_size, max_length, 1)
        #we get 1 at the last axis because we are applying score to self.V
        #the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score =self.V(tf.nn.tanh(self.W1(expanded_hidden_state), self.W2(encoder_output)))
        #now that we have a score we need to calculate the attention weights(alpha), which will just be a softmax of the score
        attention_weights = tf.nn.softmax(score, axis = 1)
        #context_vector = attention_weights*encoder_output
        context_vector = attention_weights*encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1) #what tf.reduce_sum does is explained below
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dimension, embedding_matrix, num_units, batch_size):
        super(Decoder, self).__init__()
        self.dec_units = num_units
        self.batch_size = batch_size
        self.embedding = Embedding(vocab_size, embedding_dimension, weights = [embedding_matrix], trainable = False)
        self.gru = GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fully_connected = Dense(vocab_size)
        self.attention = Attention(self.dec_units)

    def call(self, decoder_input, hidden_state, encoder_output, mode):
        #if mode = 1, will use GRU elif mode = 2 will use LSTM
        #decoder_input will be the same as decoder_output, since we are using teacher forcing method
        context_vector, attention_weights = self.attention(hidden_state, encoder_output)
        x = self.embedding(decoder_input)
        #now that we have embedding_output, we need to concatenate that with the context_vector
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        #once we put x thorugh the lstm or gru, the output will be of the shape (batch size, sequence length, units)
        if mode == 1:
            output, state = self.gru(x)
        elif mode == 2:
            output, state_c, state_h = self.lstm(x)
            state = [state_c, state_h]
        #we want the output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fully_connected(output)
        return x, state, attention_weights

def loss_function(real_output, predicted_output, loss_function_object):
    mask = tf.math.logical_not(tf.math.equal(real_output,0)) #if the actual output is 0, we dont want it equalling to 0
    loss = loss_function_object(real_output, predicted_output) #calculates the loss
    mask = tf.cast(mask, dtype = loss.dtype) #changing the dtype of mask to the dtype of loss
    loss *= mask #dunno why we are doing that

    return tf.reduce_mean(loss)
'''
training_step
1)pass input through the encoder, which will return the encoder_output and the encoder_hidden_state
2)the encoder output and its hidden_state alongside the decoder input will then be fed into the decoder, which will then return the decoder_output, decoder_hidden_state and the attention_weights
3)the decoder output is then passed into the loss function to calculate the loss
4) to get the next output, pass the decoder hidden_state from the previous timestep alongside the decoder_inputs
'''

def train(inp, output, encoder_hidden_state, encoder, decoder, batch_size, optimizer, loss_function_object):
    loss = 0
    with tf.GradientTape() as tape:
        #gradient tape allows us to calculate gradient wrt to some variables, we use it to calculate loss wrt the varibles
        encoder_output, encoder_hidden_state = encoder(inp, encoder_hidden_state, 1)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = tf.expand_dims([tokens_to_int['<start_sentence>']] * batch_size, 1) #what this line of code does is that it generates the <start> token for every training example in the batch. Explained properly below

        for i in range(1, output.shape[1]):
            predictions, decoder_hidden_state, _ = decoder(decoder_input, decoder_hidden_state, encoder_output, 1)
            loss += loss_function(output[:, i], predictions, loss_function_object)
            dec_input = tf.expand_dims(output[:, i], 1) #using teacher forcing, and setting decoder_input to the decoder_output we want
    batch_loss = loss/int(output.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradient = tape.gradient(loss, variables) #getting the gradients with respect to the variables ie. backprop
    optimizer.apply_gradients(zip(gradient, variables))
    return batch_loss


'''
explanation for tf.reduce_sum
suppose we have a (2,3) matrix of 1's, let the matrix be known as x
                    1 1 1
                    1 1 1
what tf.reduce_sum(x,0) does is --> it adds the sum of each col (ie. axis 0)
therefore tf.reduce_sum(x,0) will output --> 2 2 2

similarly if we select axis = 1, the summation will happen row wise, thus the output will be 3 3

we can also reduce the dimension alongside both the axis
ie. tf.reduce_sum(x, [0,1]) here the result will be equal to 6 (ie. 1+1+1+1+1+1)
'''
'''
decoder_input = tf.expand_dims(tokens_to_int['<start>'] * batch_size, 1) explained:
tokens_to_int['<start>'] gives us the token number of the <start> token, let it be 1
so [tokens_to_int['<start>']] will return the following ----> [1]

we need a start token for every example in the batch so we do
--> [tokens_to_int['<start>']] * batch_size
so if our batch size == 5, our code will return the following --> [1,1,1,1,1]

what we want is something like [1]
                               [1]
                               [1]
                               [1]
                               [1]
ie. a start token for each example, hence we use the tf.expand_dims([tokens_to_int['<start>']] * batch_size, 1) which returns what we want

continuing on to teacher forcing:
suppose we want the output like <start>I am fine, wbu?<end>
once we feed in the <start> token, we need to update the decoder_input so that it equals to the actual output we want from it
ie. to get "I" as the output, we need to input "I" as well if we are to use the teacher_forcing method
'''
