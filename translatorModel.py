from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


def createTokenizer(Data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(Data)
    return tokenizer


def maxLength(Data):
    return max(len(d.split()) for d in Data)
