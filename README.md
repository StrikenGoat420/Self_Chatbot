Each file and its function explained:
1) preprocessing.py 
	This file is where the data downloaded from facebook in a json is extracted and all the types of data which we will not be needing are removed, that includes images, stickers and so on. Once that was done, all the elongated words were shortened (ie. 'looool' became 'lol', 'wassssup' became 'wasup' and so on). Then we were to create a list of all common words which we would generally use, that includes common english words as well as words that were more common only in India (ie. 'ipl', 'kkr', 'kya' and so on).
	
	The idea was to then run a spell check on each of the words in our chat so that we could get the correct word as output. The spellchecker was implemented in the pycpellchecker library and was downloaded using pip. Although the spellchecker worked as it was supposed to, due to not finding a concise enough list of the words, the words were often corrected incorrectly, hence although the implementation is still there, this was not used.

2) create_embeddings.py
	Since the spellcheck did not work, another solution which I found online was to use the chars2vec embeddings ('https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d'), what it essentially does is that the word vectors of words which have similar spellings have a smaller euclidean distance between them. This although a good implementation did not assign 'meaning' to the vectors and hence was not used. Instead I used the common word2vec algorithm from the gensim library.
	
	I am currently working on a solution which can combine the advantages of both word2vec and chars2vec, the idea is that we take a threshold distance and words which have a distance less than the threshold will be deemed similar enough. Then the word2vec algorithm is applied on the words, and then the word vectors of words which have a distance less than the threshold distance in the chars2vec algorithm will be changed such that the distance between the word vectors is the same as the threshold distance. 

3) train.py
	Once we have the word embeddings and the word2vec vectors, we need to create an input and output list, this is done in this program. The inputs to the neural net is the last 5 messages concatenated, since in our conversation we talk about many things and once, and dont send just one big message, and instead send many smaller messages, this concatenation is done in this file. Once we have the concatenated inputs, we need to assign an integer for every token, this is done using the fit_on_texts function in the preprocessing_utils.py which is imported in this file. This function is similar to that present in the keras.preprocessing_utils, but instead of using the keras tokenizer, the nltk tweettokenizer was used. Similarly once we had the tokens_to_int dictionary we had to convert our inputs and outputs into sequences, this was done using the texts_to_sequences function in the preprocessing_utils.py which was imported in this file.

	The embedding_matrix was also created in this file, and all the hyperparameters and the type of loss is also defined in this file as well. Once all that is done, we have to start training and the steps for traning are as following:
	
		a)pass input through the encoder, which will return the encoder_output and the encoder_hidden_state
		b)the encoder output and its hidden_state alongside the decoder input will then be fed into the decoder, which will then return the decoder_output, decoder_hidden_state and the attention_weights
		c)the decoder output is then passed into the loss function to calculate the loss
		d) to get the next output, pass the decoder hidden_state from the previous timestep alongside the decoder_inputs

4) model.py
	The whole model is an encoder-decoder model with teacher forcing and attention.
	Since the model was an encoder-decoder model with attention, different classes was created for the Encoder, Decoder as well as the Attention mechanism. The Encoder has an Embedding layer inside it, initialized with the embedding matrix obtained from the create_embeddings.py as well as an LSTM, GRU and Bidirectional LSTM init and it is up to the user to select which one to use for encoding the text.
	
	The decoder has an embedding layer, LSTM and GRU init. The inputs for the decoder will be hidden state and the output from the encoder as well as the actual output we want from the decoder (teacher  forcing). The hidden state and the output from the encoder is then used to create the context vector (for attention) which is then concatenated with the decoder input and then fed into the LSTM/GRU. The output from the decoder is then sent into a Dense layer, which then tells us the actual prediction.

	The attention mechanism used in this is the Bahdanau Attention, and all the equations are implemented according to the paper.

5) preprocessing_utils.py
	This file contains almost all the helper functions used in all of the other files alongside comments for what each helper function does.


	
