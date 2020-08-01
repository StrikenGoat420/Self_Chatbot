from preprocessing_utils import *
from dict import dictionary

source_fileloc = r'/home/shubham/Python/project/chatbot/messages/inbox/dharanimathivanan_ncbmlzxsww'
source_filenames = ['message_1.json', 'message_2.json', 'message_3.json', 'message_4.json', 'message_5.json', 'message_6.json', 'message_7.json']
raw_messages = []
nm = [] #hack to write the messages into a csv file

for i in range(6,-1,-1):
    file_loc = source_fileloc+"/"+source_filenames[i]
    open_file(file_loc, raw_messages)

remove_contentless_message(raw_messages)
merge(raw_messages, nm)
write_csv(nm, 'chats.csv', True)

data_loc = r'/home/shubham/Python/project/chatbot/chats.csv'
dict_loc = r'/home/shubham/Python/project/chatbot/dictionary_stuff/google-10000-english/20k.txt'
slangs_loc = r'/home/shubham/Python/project/chatbot/dictionary_stuff/slangs.csv'

data = [] #list of data
dict = [] #list of words in dictionary

open_file(data_loc, data)
open_file(dict_loc, dict)
open_file(slangs_loc, dict)
dict = np.array(dict)
dict = list(dict.flatten())
dict += dictionary


processed_data = []
word_count = {}
TOKENS = [] #dont need this outside the function, but it maybe useful to later on if we need to check how the tokenizer performs

remove_zeros(r'/home/shubham/Python/project/chatbot/clean_data1.csv')
'''the following commented functions do work properly, but due to the large dictionary size most words are corrected into some unintended words according to the levenstein distance, hence instead of using autocorrect to get the
    correct word, we'll be implementing a character level model, with some basic text preprocessing such as all 'hahaha.*' becomes 'hahaha' and so on this is implemented in the remove_zeros function in the utils file, out of sheer laziness'''
#normalise_and_spellcorrect(data, dict, word_count, processed_data, TOKENS)
#write_csv(processed_data, 'clean_data.csv', False)
