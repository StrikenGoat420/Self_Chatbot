How this program works:

	1)Download and preprocess the data from facebook (The preprocessing is done in the preprocessing.py)
	
	2)Once that is done, create the embedding for each word (This is done in the create_embedding.py)
		a)I tried various different approach over here including using a chars2vec model I found online, this was necessary because there are alot of typos/elongated words present in our chats. I have an idea where I can make full use of the advantages both word2vec and char2vec provide and will be implementing that asap.
		
	3)Still some preprocessing left (this is done in the pre_train.py, but should have been done in the preprocessing.py), in this file an embedding matrix is created, all the sentences are converted into a list of integers which are to be fed into the actual model.
	
	4)The actual model and is in model.py and I'm still working on it.
	

Almost all of the functions I have used or thought would be useful are in preprocessing_utils.py, this includes a custom version of the 'fit_on_texts' and the 'texts_to_sequences' function which is present in keras, the difference between the ones I made and the one present in Keras is the Tokenizer. I used the TweetTokenizer from nltk as it was just more appropriate for my usecase. The most important/useful file for anybody is the preprocessing_utils.py


PS. I know the code is very sloppy, but this is just a small project of mine so bear with me
