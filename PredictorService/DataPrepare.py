import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string


class DataPrepare:
    def __init__(self):
        vocab_size = 50000
        batch_size = 128
        maxlen = 5
        filenames = ["test.txt"]
        self.text_ds = tf.data.TextLineDataset(filenames)
        self.text_ds = self.text_ds.shuffle(buffer_size=256)
        self.text_ds = self.text_ds.batch(batch_size)
        self.vectorize_layer = TextVectorization(
            standardize=self.custom_standardization,
            max_tokens=vocab_size - 1,
            output_mode="int",
            output_sequence_length=maxlen + 1,
        )
        self.vectorize_layer.adapt(self.text_ds)
        self.vocab = self.vectorize_layer.get_vocabulary()  # To get words back from token indices
        self.text_ds = self.text_ds.map(self.prepare_lm_inputs_labels)
        self.text_ds = self.text_ds.prefetch(tf.data.experimental.AUTOTUNE)

    def custom_standardization(self, input_string):
        """ Remove html line-break tags and handle punctuation """
        lowercased = tf.strings.lower(input_string)
        stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
        return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")

    def prepare_lm_inputs_labels(self, text):
        """
        Shift word sequences by 1 position so that the target for position (i) is
        word at position (i+1). The model will use all words up till position (i)
        to predict the next word.
        """
        text = tf.expand_dims(text, -1)
        tokenized_sentences = self.vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y

    def get_dataset(self):
        return self.text_ds

    def get_vocab(self):
        return self.vocab

    def get_vectorize_layer(self):
        return self.vectorize_layer

