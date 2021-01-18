from gensim.models import KeyedVectors
from PredictorService.MultyHead import *
from PredictorService.DataPrepare import *
from PredictorService.TextGenerator import *


class NewNeuralNet:

    def __init__(self):
        self.w2v_model = KeyedVectors.load("word_to_vec.wordvectors", mmap='r')
        self.vocab_size = 50000  # Only consider the top 20k words
        self.maxlen = 5
        self.embed_dim = 256  # Embedding size for each token
        self.num_heads = 2  # Number of attention heads
        self.ff_dim = 256  # Hidden layer size in feed forward network inside transformer
        self.batch_size = 128
        self.model = self.create_model()
        try:
            self.model.load_weights("weights.h5")
        except:
            print("Load weights error")
        print("init completed")

    def create_model(self):
        inputs = layers.Input(shape=(self.maxlen,), dtype=tf.int32)
        embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        x = transformer_block(x)
        outputs = layers.Dense(self.vocab_size)(x)
        model = keras.Model(inputs=inputs, outputs=[outputs, x])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(
            "adam", loss=[loss_fn, None],
        )  # No loss and optimization based on word embeddings from transformer block
        return model

    def prepare_data(self):
        preparer = DataPrepare()
        self.data = preparer.get_dataset()
        self.vocab = preparer.get_vocab()
        self.vectorize_layer = preparer.get_vectorize_layer()

    def fit_model(self, epochs):
        word_to_index = {}
        for index, word in enumerate(self.vocab):
            word_to_index[word] = index
        start_prompt = "у меня сосед бабка"
        start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
        num_tokens_generated = 40
        text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, self.vocab)
        self.model.fit(self.data, verbose=1, epochs=epochs, callbacks=[text_gen_callback])

    def predict_words(self, text):
        text = tf.expand_dims(text, -1)
        vector_for_predict = self.vectorize_layer(text)
        vector_for_predict = vector_for_predict[:, :-1]
        predict = self.model.predict(vector_for_predict)
        predict = self.vocab[predict[0][0][0].argmax()]
        try:
            predict = self.w2v_model.most_similar(predict, topn=3)
        except:
            pass
        return predict

    # def split_sentence(self, data):
    #     X = np.array([data[i:i+3] for i in range(2831751)])
    #     #X = np.array([data[i:i+3] for i in range(2831755)])
    #     y = np.array([data[i] for i in range(3, 2831751)])
    #     #y = to_categorical(y, num_classes=20000)
    #     x = np.zeros((2831751, 3))
    #     for i in range(2831751):
    #         x[i] = X[i]
    #     print("split completed")
    #     return [x, y]



    # def create_learning_data(self):
    #     with open('test.txt', 'r', encoding='utf-8') as f:
    #         texts = f.read()
    #         texts = texts.replace('\ufeff', '')
    #     maxWordsCount = 20000
    #     self.tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
    #                               lower=True, split=' ', char_level=False)
    #     self.tokenizer.fit_on_texts([texts])
    #     self.dataqwe = self.tokenizer.texts_to_sequences([texts])
    #     (X, y) = self.split_sentence(self.dataqwe[0])
    #     y = self.convert_y(y)
    #     print("create data completed")
    #     return [X, y]
    #
    # def convert_y(self, data):
    #     Y = list()
    #     for i in range(data.size - 1):
    #         Y.append(self.tokenizer.index_word[data[i]])
    #     yp = np.array(Y)
    #     arr = np.zeros(shape=(np.shape(yp)[0], 100))
    #     print(arr.shape)
    #     for i in range(np.shape(yp)[0]):
    #         try:
    #             arr[i] = np.array([self.w2v_model.wv[yp[i]]])
    #         except:
    #             arr[i] = np.array([np.zeros(shape=(100))])
    #     return arr

    # def main(self):
    #     ann = NewNeuralNet()
    #     (X_train, y) = ann.create_learning_data()
    #     Y = list()
    #     for i in range(y.size):
    #         Y.append(self.tokenizer.index_word[y[i]])
    #     Y = np.array(Y)
    #     y_train = ann.convert_y(Y)
    #     self.model.compile("nadam", loss="log_cosh", metrics=['accuracy'])
    #     history = self.model.fit(X_train[:2831748], y_train[:2831748], epochs=1)
