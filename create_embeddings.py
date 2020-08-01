'''input csv data, and create word embeddings of given dimensions to be used later'''
'''
import clean csv file, combine all the text in it into a single text file (or string) - for skip gram word2vec model - once done create input and output features and feed it into a model
'''
import numpy as np
#import chars2vec
import csv
from preprocessing_utils import open_file
from nltk.tokenize import TweetTokenizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


def get_token(messages, tokens):
    '''function takes in a list of messages as inputs and returns a list of all tokens as output '''
    tokenizer = TweetTokenizer()
    messages = np.array(messages)
    messages = list(messages.flatten())
    for m in messages:
        tokens += tokenizer.tokenize(m)



def get_sentence_tokens(messages, sentence):
    '''takes in a list of sentences, and converts each sentence into a list of tokens. Eg - ["This is a sentence"] becomes ["This",'is','a','sentence']'''
    tokenizer = TweetTokenizer()
    messages = np.array(messages)
    for i in range(len(messages)):
        #the line below does the following [[a,b],[c,d]] --> [[a b], [c d]]
        sent = ' '.join(messages[i])
        token = tokenizer.tokenize(sent)
        sentence.append(token)



messages = []
tokens = []
file_loc = r'/home/shubham/Python/project/chatbot/cleaned_data.csv'
open_file(file_loc, messages)
#get_token(messages, tokens)
sentence = []
get_sentence_tokens(messages, sentence)

print(f'total number of sentences is {len(sentence)}')

print(1)
model = Word2Vec(sentence, min_count=1, iter= 25000, sg=1, window=10, size=300)
#model.save("word2vec3.model")
print(2)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

'''
word_to_vector = {}
words = []

for i in range(len(tokens)):
    if tokens[i] not in word_to_vector:
        word_to_vector[tokens[i]] = 0
        words.append(tokens[i])

c2v_model = chars2vec.load_model('eng_100')

word_embeddings = c2v_model.vectorize_words(words)
word_to_vector = {}
for i in range(len(words)):
    word_to_vector[words[i]] = word_embeddings[i]
'''








'''
words = ['Language', 'Natural', 'Understanding',
         'Naturael', 'Longuge', 'Understanding',
         'Motural', 'Lamnguoge', 'Understaating',
         'Naturrow', 'Laguage', 'Unddertandink',
         'Nattural', 'Languagge', 'Umderstoneding']

word_embeddings = StandardScaler().fit_transform(word_embeddings)
projection_2d = sklearn.decomposition.PCA(n_components=3).fit_transform(word_embeddings)

# Draw words on plane
f = plt.figure(figsize=(28, 26))
for j in range(len(projection_2d)):
    try:
        plt.scatter(projection_2d[j, 0], projection_2d[j, 1],
                    marker=('$' + words[j] + '$'),
                    s=500 * len(words[j]), label=j,
                    facecolors='green' if words[j]
                               in ['Natural', 'Language', 'Understanding'] else 'black')
    except ValueError:
        pppp = 1
plt.show()

'''
