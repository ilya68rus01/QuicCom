import re
import nltk
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


class PredictorService:
    def __init__(self):
        nltk.download('stopwords')
        self.patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
        self.stopwords_ru = stopwords.words("russian")
        self.morph = MorphAnalyzer()
        self.create_ann()
        self.ann_model.load_weights("big_model.h5")
        print("prepare data")
        #self.__preparing_data__()
        print("data prepared")


    def __preparing_data__(self):
        data_fr = pd.read_csv('dataframe.csv', delimiter=';')
        self.data_frame = pd.Series(data_fr['Comment'])
        self.data_frame = self.data_frame.dropna().drop_duplicates()
        self.data_frame = self.data_frame.apply(self.__lemmatize__)
        self.data_frame = self.data_frame.dropna()

    def __lemmatize__(self, doc):
        doc = re.sub(self.patterns, ' ', doc)
        tokens = []
        for token in doc.split():
            if token and token not in self.stopwords_ru:
                token = token.strip()
                token = self.morph.normal_forms(token)[0]
                tokens.append(token)
        if len(tokens) > 2:
            return tokens
        return None

    def main(self):
        try:
            self.w2v_model = KeyedVectors.load("word2vec.wordvectors", mmap='r')
            print("w2v load")
        except:
            print("error")
            self.__create_w2v_model__()
        #self.__load_ann__()
        print("ann load")

    def convert_to_vec(self):
        try:
            self.w2v_model.load("word2vec.wordvectors")
        except:
            self.__create_w2v_model__()
        (X, y) = self.split_sentence(self.data_frame)
        X_all = self.convert_x(X)
        y_all = self.convert_y(y)
        self.create_ann()
        train_info = self.ann_model.fit(X_all, y_all, epochs=150, verbose=1)

    def convert_x(self, data):
        arr = np.zeros(shape=(np.shape(data)[0], 3, 70))
        for i in range(np.shape(data)[0]):
            try:
                arr[i] = np.array([self.w2v_model.wv[data[i]]])
            except:
                arr[i] = np.array([np.zeros(shape=(3, 70))])
        return arr

    def convert_y(self, data):
        arr = np.zeros(shape=(np.shape(data)[0], 70))
        for i in range(np.shape(data)[0]):
            try:
                arr[i] = np.array([self.w2v_model.wv[data[i]]])
            except:
                arr[i] = np.array([np.zeros(shape=(70))])
        return arr

    def split_sentence(self, data):
        x = list()
        lst_x = list()
        lst_y = list()
        for words in data:
            i = 0
            z = 3
            lst_x = list()
            while i < len(words):
                if i == z:
                    lst_y.append(words[i])
                    i = i - 2
                    z += 1
                    x.append(lst_x)
                    lst_x = list()
                lst_x.append(words[i])
                i += 1
        return [x, lst_y]

    def create_ann(self):
        self.ann_model = keras.Sequential()
        self.ann_model.add(layers.Input(shape=(3, 70)))
        self.ann_model.add(keras.layers.BatchNormalization())
        self.ann_model.add(layers.Dropout(0.2))
        self.ann_model.add(layers.Dense(500, activation='sigmoid'))
        self.ann_model.add(layers.Dropout(0.2))
        self.ann_model.add(keras.layers.BatchNormalization())
        self.ann_model.add(layers.Dense(210, activation='sigmoid'))
        self.ann_model.add(keras.layers.BatchNormalization())
        self.ann_model.add(layers.LSTM(64))
        self.ann_model.add(layers.Dense(150, activation='sigmoid'))
        self.ann_model.add(layers.Dropout(0.2))
        self.ann_model.add(keras.layers.BatchNormalization())
        self.ann_model.add(layers.Dense(70, activation='tanh'))
        self.ann_model.compile(loss='mean_squared_error', optimizer='nadam',
                               metrics=[tf.keras.metrics.RootMeanSquaredError()])

    def __create_w2v_model__(self):
        self.w2v_model = Word2Vec(
            min_count=3,
            window=3,
            size=70,
            negative=10,
            alpha=0.03,
            min_alpha=0.0007,
            sample=6e-5,
            sg=1)
        self.w2v_model.build_vocab(self.data_frame)
        self.w2v_model.train(self.data_frame, total_examples=self.w2v_model.corpus_count,
                             epochs=300, report_delay=1)

    def batch_train_ann_model(self, data):
        sentence = self.preparing_data_for_predict(data)
        (X, y) = self.split_sentence(sentence)
        X = self.convert_x(X)
        y = self.convert_y(y)
        self.ann_model.train_on_batch(X, y)

    def predict_next_word(self, data):
        sentence = self.preparing_data_for_predict(data)
        (x, y) = self.split_sentence(sentence)
        x = self.convert_x(x)
        y = self.convert_y(y)
        try:
            pred = self.ann_model.predict(x)
            for vec in pred:
                next_possible_words = self.w2v_model.wv.similar_by_vector(vec, topn=5)
            lst = list()
            for word in next_possible_words:
                lst.append(word[0])
        except:
            lst = list()
        return lst

    def preparing_data_for_predict(self, sentence):
        data_frame = pd.Series(sentence)
        data_frame = data_frame.dropna().drop_duplicates()
        data_frame = data_frame.apply(self.__lemmatize__)
        data_frame = data_frame.dropna()
        return data_frame

    def __load_ann__(self):
        try:
            self.ann_model = keras.models.Sequential()
            self.ann_model = load_model("big_model.h5")
        except:
            print("File load error")
