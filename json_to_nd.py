import gensim
import sys
import json
import numpy as np
import pickle
import os

if not len(sys.argv) == 2:
    sys.exit('Invalid command line arguments! Enter expected JSON file')

GOOGLE_NEWS_MODEL_PATH = 'GoogleNews-vectors-negative300.bin'

model = gensim.models.Word2Vec.load_word2vec_format(GOOGLE_NEWS_MODEL_PATH, binary=True)
basename = os.path.basename(sys.argv[1])
dump_file_x = open(os.path.splitext(basename)[0] + '_x.txt','wb')
dump_file_y = open(os.path.splitext(basename)[0] + '_y.txt','wb')

with open(sys.argv[1], 'r') as json_file:
    for line in json_file:
        json_obj = json.loads(line)
        review_text = json_obj['reviewText']
        summary = json_obj['summary']
        review_text_vec = []
        summary_vec = []
        for word in review_text.split():
            try:
                word_as_vec = model[word]
                if len(review_text_vec) == 0:
                    review_text_vec = np.transpose(np.reshape(word_as_vec, (np.shape(word_as_vec)[0], 1)))
                else:
                    review_text_vec = np.concatenate((review_text_vec, np.transpose(np.reshape(word_as_vec, (np.shape(word_as_vec)[0], 1)))), axis = 0)
            except:
                pass
        print(review_text_vec.shape)
        pickle.dump(review_text_vec, dump_file_x)
        for word in summary.split():
            try:
                word_as_vec = model[word]
                if len(summary_vec) == 0:
                    summary_vec = np.transpose(np.reshape(word_as_vec, (np.shape(word_as_vec)[0], 1)))
                else:
                    summary_vec = np.concatenate((summary_vec, np.transpose(np.reshape(word_as_vec, (np.shape(word_as_vec)[0], 1)))), axis = 0)
            except:
                pass
        print(review_text_vec.shape)
        pickle.dump(summary_vec, dump_file_y)
    dump_file_x.close()
    dump_file_y.close()