'''this is the python file where the model is created and then trained'''
import csv
import numpy as np
from preprocessing_utils import open_file, fit_on_texts, texts_to_sequences
from preprocessing_utils import merge_inputs
from gensim.models import Word2Vec, KeyedVectors
from keras.preprocessing.sequence import pad_sequences
#from try import fit_on_texts

word_vectors = KeyedVectors.load(r'word2vec.kv', mmap='r')

def get_inputs_and_outputs(messages, inputs, outputs):
    for m in messages:
        for i in range(len(m)):
            if i == 0:
                inputs.append(m[i])
            else :
                outputs.append(m[i])

messages = []
open_file(r'/home/shubham/Python/project/chatbot/cleaned_data.csv', messages)
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

for i in range(len(X)):
    X[i] = "<start_sentence> "+X[i]+" <end_sentence>"
    Y[i] = "<start_sentence> "+Y[i]+" <end_sentence>"

texts_to_sequences(X, tokens_to_int, encoded_input)
texts_to_sequences(Y, tokens_to_int, encoded_output)



max_length = 0
idx = 0
count = 0
'''this for loop is to get the maximum length of input so that we can pad accordingly'''
for i in range(len(encoded_input)):
    if len(encoded_input[i]) > 150:
        #just for myself
        count += 1
    if len(encoded_input[i]) > max_length:
        max_length = len(encoded_input[i])
        #idx is to keep track of the index of the longest sentence, for personal use
        idx = i
#print(f"max len {max_length} idx is {idx} count is {count}")
#print(X[idx])
#max_length = 100

padded_inputs = pad_sequences(encoded_input, maxlen=max_length, padding='post')
padded_outputs = pad_sequences(encoded_output, maxlen=max_length, padding='post')

vocab_size = len(tokens_to_int) + 1
embedding_dimension = word_vectors['sup'].shape[0]


embedding_matrix = np.zeros((vocab_size, embedding_dimension))
#print(embedding_matrix.shape)

i = 0
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

    #if np.array_equal(embedding_matrix[i],word_vectors[keys]):  # test if same shape, same elements values
    #    print('yes')

'''not creating testing set'''
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
#print(f"train len {len(X_train)} {len(Y_train)}")
#print(f"test len {len(X_test)} {len(Y_test)}")
