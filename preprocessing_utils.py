'''file containing list of all Preprocessing functions'''
'''explanation comments at the bottom'''
'''file now contains a list of all functions deemed useful'''


import csv
import json
import string
import numpy as np
from spellchecker import SpellChecker
import re
from nltk.corpus import words
from nltk.tokenize import TweetTokenizer
import time
from itertools import islice
import glob
import os
from gensim.models import Word2Vec, KeyedVectors

def test_open(file_loc, messages):
    '''not using this method because it reads the json files in random'''
    for filename in glob.glob(os.path.join(file_loc, '*.json')):
        print(f"file name is {filename}")
        with open(filename) as f: # open in readonly mode
            data = json.load(f)
            message = data['messages']
            message.reverse()
            messages.extend(message)
      # do your stuff

def open_file(file_loc, messages):
    if file_loc.endswith('.json'):
        f = open(file_loc)
        data = json.load(f)
        message = data['messages']
        message.reverse()
        messages.extend(message)
        f.close()
    else:
        f = open(file_loc)
        reader = csv.reader(f)
        messages += list(reader)
        f.close()

def remove_contentless_message(messages):
    '''to remove messages which do not have any content in them'''
    alphanumeric = list(string.ascii_lowercase) + list(string.ascii_uppercase) + list(string.digits)
    mwith_no_content = []
    i = 0
    for m in messages:
        i+= 1
        try :
            if m['content'][0] not in alphanumeric:
                messages.remove(m)
        except KeyError:
            messages.remove(m)
            mwith_no_content.append(i) #to keep track of id of messages which do not have content

def write_csv(messages, name, write_chats_from_json):
    OutFile = open(name, 'w', newline = '')
    writer = csv.writer(OutFile)
    if write_chats_from_json == True:
        k = 0
        for i in range(len(messages)):
            #a is the difference between the current and the previous timestamp in ms
            #a = int(messages[i]['timestamp_ms']) - int(messages[i-1]['timestamp_ms'])
            #b is the difference between the current and the previous timestamp in minutes
            #b = a/60000
            #c is the same difference in hours
            #c = b/60
            try:
                if messages[i]['sender_name'] == 'Shubham Pareek':
                    #writer.writerow([messages[i]['content'], 0, messages[i]['timestamp_ms'],b,c])
                    writer.writerow([messages[i]['content'], 0])
                else :
                    #writer.writerow([0,messages[i]['content'], messages[i]['timestamp_ms'],b,c])
                    writer.writerow([0,messages[i]['content']])
            except KeyError:
                k += 1
                #print(k)
    else:
        for row in messages:
            writer.writerow(row)
        #it = iter(messages)
        #n = 2 #n is the number of slices you want to make of the list, for better understanding go to https://stackoverflow.com/a/17483635
        #processed_list = [list(islice(it,n)) for _ in range(int(len(messages)/n))]
        #for row in processed_list:
        #    writer.writerow(row)

    OutFile.close()


def merge(messages, nm):
    outline = {'sender_name':'', 'content':''}
    j = 1
    for i in range(len(messages)):
        try:
            if messages[i]['sender_name'] == messages[i-1]['sender_name']:
                '''merge contents'''
                messages[i]['content'] = messages[i-1]['content'] +' '+messages[i]['content']
            else :
                '''stop merging'''
                outline['sender_name'], outline['content'] = messages[i-1]['sender_name'], messages[i-1]['content']
                nm.append(outline)
                outline = {'sender_name':'', 'content':''}

        except (KeyError, IndexError) as e:
            a = 1

def remove_consecutive_repeating_chars(data):
    regex = r"(.)\1+"
    #loop to remove all consecutive duplicate characters in a word
    for i in range(len(data)):
        #print(data[i])
        for j in range(len(data[i])):
             data[i][j] = re.sub(regex, r'\1', data[i][j], 0, re.MULTILINE)

def normalise_and_spellcorrect(data, dict, word_count, processed_data, TOKENS):
    tokenizer = TweetTokenizer()

    remove_consecutive_repeating_chars(data)

    spell = SpellChecker(language = '', case_sensitive=True)
    a = spell.word_frequency.load_words(dict)
    #range will be range(len(data))
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == '0':
                continue
            tokens = tokenizer.tokenize(data[i][j])
            TOKENS.append(tokens)

    for i in range(len(TOKENS)):
        for j in range(len(TOKENS[i])):
            TOKENS[i][j] = TOKENS[i][j].lower()
            if 'hahaha' in TOKENS[i][j]:
                print(str(i) + ' got into first if : ' + TOKENS[i][j])
                TOKENS[i][j] = 'hahaha'
            elif len(TOKENS[i][j]) > 10 or TOKENS[i][j] in dict:
                print(str(i) + " got into the 'continue' if : " + TOKENS[i][j])
                continue
            else:
                print(str(i) + ' got into spellcheck if : ' + TOKENS[i][j])
                TOKENS[i][j] = spell.correction(TOKENS[i][j])
            if TOKENS[i][j] not in word_count:
                word_count[TOKENS[i][j]] = 1
            else:
                word_count[TOKENS[i][j]] += 1
        clean_data = ' '.join(TOKENS[i])
        processed_data.append(clean_data)

def remove_zeros(file_loc, file_name):
    '''function receives the location of the chats file which is in the format of [[a,0][0,b][c,0][0,d]] and returns a csv file in the given format [[a,b][c,d]]'''
    f = open(file_loc)
    data = list(csv.reader(f))
    #print(data)

    for i in range(len(data) -1 ):
        for j in range(len(data[i])):
            data[i][j] = data[i][j].lower()
            if data[i][j] == '0' and j+1%2 == 2:
                data[i][j] = data[i+1][j]
    new_data = []
    for i in range(0, len(data),2):
        new_data.append(data[i])
    '''to remove all repeating letters'''
    tokenizer = TweetTokenizer()
    for i in range(len(new_data)):
        for j in range(len(new_data[i])):
            tokens = tokenizer.tokenize(new_data[i][j])
            for k in range(len(tokens)):
                if 'hahaha' in tokens[k]:
                    tokens[k] = 'hahaha'
                elif 'lol' in tokens[k]:
                    tokens[k] = 'lol'
            new_data[i][j] = ' '.join(tokens)
    write_csv(new_data, file_name, False)

def merge_inputs(messages, output, N):
    '''This function merges the last 5 messages we sent to each other as one big input. This is because many times in our conversation we talked about multiple topics at once
        Input a list of the following type [[a,b], [c,d], [e,f], [g,h], [i,j], [k,l], [m,n], [o,p], [q,r]]
                Merge the last n=5 messages into a single input
                eg --> above list becomes [[[a],b], [[a,b,c], d], [[a,b,c,d,e], f], [[c,d,e,f,g], h], [[e,f,g,h,i], j], [[g,h,i,j,k], l] .....]
        In the second for loop we use the range(2,-1,-1), we use 2 because each input from the messages contains one prompt and one reply - a total of 2 messages -
        # TODO: Come up with a way where I dont have to hardcode the 2 in the for loop, and use the N which the user inputs to come up with the range'''
    for i in range(len(messages)):
        inp = ''
        out = ''
        out_list = []
        out_list1 = []
        for n in range(2,-1,-1):
            if i-n < 0:
                continue
            #print(f"i is {i} n is {n} and i-n is {i-n}")
            if i-n >= 0:
                #print(f"inside first if and A[i-n] is {A[i-n]}")
                if n > 0:
                    #print(1, A[i-n])
                    inp = inp + ' '+ ' '.join(messages[i-n])
                if n == 0:
                    #print(2, A[i-n])
                    inp += " " +messages[i-n][0]
                    out = messages[i-n][1]
                #print(i, A[i-n])
        out_list.append(inp)
        out_list1.append(out_list)
        out_list1.append(out)
        output.append(out_list1)
        #print(i, inp)
        #print('------')


def fit_on_texts(file_loc, token_to_int):
    '''this function does the same thing as keras.Tokenizer.fit_on_texts(), but instead uses the TweetTokenizer from nltk to get better tokens'''
    '''it takes in a pretrained word2vec.kv file and returns a dictionary of all the tokens in it with an int assigned to each token'''
    #input dictionary should be empty
    wv = KeyedVectors.load(file_loc)
    i = 1
    for word, vocab_obj in wv.vocab.items():
        if word not in token_to_int:
            token_to_int[word] = i
        i += 1

def texts_to_sequences(sentences, token_to_int, output_list):
    '''takes in a list of sentences and changes them to a list to of integers from the token_dict'''
    tokenizer = TweetTokenizer()
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        sequence = []
        for tok in tokens:
            sequence.append(token_to_int[tok])
        output_list.append(sequence)

'''the following functions and comments are not of any use as such, and were used in trying out various things'''

'''
TODO: Find a proper way to multiply the embedding matrix with the character representation generated from the char2vec, to get make maximum use of the vectors generated from chars2vec
Ideas so far: 1)Do an element wise multplication of the chars2vec representation with the appropriate coloumn in the Embedding matrix
              2)If distance between two words is below a particular threshold, just take the average and instead of the '1' in the one hot encoding vector, put the average (Below progman was to check the distance between two words)
'''

def find_dist_for_each_word(word2vec, words, dist_for_sim_word):
    '''dist_for_sim_word is a boolean, which if true will distance only for words that are similar to each other in the word list. Only useful in the test file'''
    for i in range(0, len(words)-1):
        for j in range(i+1, len(words)):
            if dist_for_sim_word == True:
                if (j-i)%3 == 0:
                    find_distance1(word1 = words[i], word2 = words[j], word_to_vector = word2vec)
            else:
                if (j-i)%3 != 0:
                    find_distance1(word1 = words[i], word2 = words[j], word_to_vector = word2vec)

def find_distance1(**kwargs):
    '''function to find the euclidean distance between two word vector'''
    '''not a good implementation of kwargs'''
    a = 0
    if 'word1' not in kwargs:
        mat1 = kwargs['mat1']
        mat2 = kwargs['mat2']
    else:
        word1 = kwargs['word1']
        word2 = kwargs['word2']
        mat1 = kwargs['word_to_vector'][word1]
        mat2 = kwargs['word_to_vector'][word2]
        a = 1
    #below line is euclidean distance
    dist = np.linalg.norm(mat1 - mat2)
    if dist < 1:
        if a == 1:
            print(f"distance between {word1} and {word2} is {dist}")
        else:
            print(f"distance between mat1 and mat2 is {dist}")

def reshape_embeddings(old_embeddings, new_embeddings):
    #function to reshape vectors os shape (n, ) to shape (n,1)
    #these vectors are inside the old_embeddings array
    for i in range(len(old_embeddings)):
        #print(word_embeddings[i].shape)
        new_embedding = word_embeddings[i].reshape((100,1))
        new_embeddings.append(new_embedding)
    new_embeddings = np.array(new_embeddings)

def create_one_hot_encoding(words, encoding_list):
    for i in range(len(words)):
        one_hot_emb = []
        for j in range(len(words)):
            if j == i:
                one_hot_emb.append(1)
            else:
                one_hot_emb.append(0)
        one_hot = np.array(one_hot_emb).reshape((len(words), 1))
        encoding_list.append(one_hot)

'''
TODO :Preprocessing steps:
        Import data in csv format
        Tokenize each word
        Clean words as much as possible, run spellcheck on words to correct the spellings
        Shorten elongated words, ie. 'duuuuuuudeee' becomes 'dude'

     Embeddings steps:
        Once text has been preprocessed use n-grams to create word embeddings
'''

'''
correcting elongated words:
    1) Remove all charcaters that have been repeated multiple times, ie - haaaaappyyyyy becomes happy, lolllllll becomes lol, wttttttfffff becomes wtf and so on
    2) Create a dictionary of english words (get from nltk) and add all popular english slangs and some emotes(eg, :p) in the dictionary as well
    3) Use spellcheck to find missplled words and correct them, ie - hapy becomes happy
'''
