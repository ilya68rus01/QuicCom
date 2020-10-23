import re
import nltk
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec

class PredictorService:
    def __init__(self):
        nltk.download('stopwords')
        self.patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
        self.stopwords_ru = stopwords.words("russian")
        self.morph = MorphAnalyzer()
        self.data_frame = pd.read_csv("../lemm.csv", delimiter=';')
        self.__create_model__()

    def __create_model__(self):
        Word2Vec(sentences=self.data_frame)
        self.w2v_model = Word2Vec(
            min_count=10,
            window=2,
            size=300,
            negative=10,
            alpha=0.03,
            min_alpha=0.0007,
            sample=6e-5,
            sg=1)
        self.w2v_model.build_vocab(self.data_frame)
        self.w2v_model.train(self.data_frame, total_examples=self.w2v_model.corpus_count, epochs=120, report_delay=1,
                             compute_loss=True)
        # print(w2v_model.wv.most_similar(positive=["любить"]))
        # print(w2v_model.wv.word_vec('конец', 'любить').shape)
        #self.convert_sentence_in_list_vec("Любить конец в принципе")
        #print(self.w2v_model.wv['смерть'])
        # 'конец', 'любить'

    def close_words(self, word):
        try:
            most_sim_vec = self.w2v_model.wv.most_similar(positive=word, topn=5)
            answer = "Наиболее близкие слова: "
            for words in most_sim_vec:
                answer += words[0] + " "
        except:
            answer = "Я не знаю этого слова :("
        return answer