from pre_train import embedding_matrix, vocab_size, embedding_dimension, padded_inputs, padded_outputs
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten,Input, BatchNormalization, LSTM, Attention, Embedding, Bidirectional
import numpy as np
#print(padded_inputs.shape)

'''
    after preprocessing my inputs and outputs are in the following format
    <start_sentence> some message <end_sentence>, this is then converted to a list of ints according to the tokens_to_int dictionary
'''

input_shape = (padded_inputs.shape[1], )
#print(padded_inputs[0])
#print(padded_outputs[0])

def get_model(input_shape, output_shape, num_units):
    #input shape is a tuple, and so is output shape
    encoder_inputs = Input(input_shape)
    embedding_layer = Embedding(vocab_size, embedding_dimension, weights = [embedding_matrix], input_length=input_shape[0], trainable = False)

    encoder_embedding_layer = embedding_layer(encoder_inputs)
    normalized_inputs = BatchNormalization()(encoder_embedding_layer)
    lstm_encoder_layer = LSTM(num_units, return_state=True, return_sequences=True)
    encoder = Bidirectional(lstm_encoder_layer, merge_mode='concat')
    encoder_outputs, state_h, state_c = lstm_encoder_layer(normalized_inputs)#return sequences is set to false cuz we dont need the sequence
    #state_h, state_c = encoder(encoder_embedding_layer)#return sequences is set to false cuz we dont need the sequence
    encoder_states = [state_h, state_c]
    #works till here
    '''
    # TODO: take in decoder inputs which will be the same as the correct decoder output in the previous time step
            create an attention module which takes in the encoder states at each time step, and generates the context which will then be used as the inital state of the decoder
            study up on attention
    '''
    decoder_inputs = Input(input_shape)
    decoder_embedding_layer = embedding_layer(decoder_inputs)
    normalized_outputs = BatchNormalization()(decoder_embedding_layer)
    decoder = LSTM(num_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _= decoder(normalized_outputs, initial_state = encoder_states)
    decoder_dense = Dense(input_shape[0], activation='softmax')
    normalized_outputs = BatchNormalization()(decoder_outputs)
    decoder_outputs = decoder_dense(normalized_outputs)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])
    #model = Model(inputs=[embedder_inputs], outputs=[embedding_layer])#this was used to check if the embedding layer worked properly or not
    return model


model = get_model(input_shape, (220,120), 128)
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit([padded_inputs, padded_outputs], [padded_outputs],epochs=100, validation_split=0.2)


#ped = padded_inputs[:2]
#predictions = model.predict(ped)
#print(predictions)




'''
have padded_inputs, padded_outputs
# TODO: 1)feed in the padded inputs into the embedding layers
        2)get the embedding representation, then feed in that representation in a bidirectional lstm
        3)feed in the outputs of the bidirectional lstm into the attention module to get context
        4)feed in the context into the post attention unidirectional lstm module to get the outputs
'''
