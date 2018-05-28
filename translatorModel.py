from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Activation


from pickle import load
import numpy as np


def createTokenizer(Data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(Data)
    return tokenizer


def maxLength(Data):
    return max(len(d.split()) for d in Data)


def loadLanguageFile(filename):
    languageFile = load(open(filename, 'rb'))
    return languageFile


def encodeSequences(trainingData, tokenizer, maxlength):
    X = []
    for sequence in trainingData:
        encoder = tokenizer.texts_to_sequences([sequence])
        Seq = pad_sequences([encoder], maxlen=maxlength, padding='pre')
        X.append(Seq)
    return X


def encodeOutput(testData, vocabSize):
    y = []
    for sequence in testData:
        Seq = to_categorical(sequence, num_classes=vocabSize)
        y.append(Seq)
    y = np.array(y)
    return y


english = loadLanguageFile('english.pkl')
french = loadLanguageFile('french.pkl')


trainingSize = 120000
trainEng = english[:trainingSize]
trainFr = french[:trainingSize]
testEng = english[:(len(english) - trainingSize)]
testFr = french[:(len(french) - trainingSize)]
englishTokenizer = createTokenizer(trainEng)
frenchTokenizer = createTokenizer(trainFr)
englishVocabSize = len(englishTokenizer.word_index) + 1
frenchVocabsize = len(frenchTokenizer.word_index) + 1
englishMaxlength = maxLength(trainEng)
frenchMaxLength = maxLength(trainFr)


print("Dataset size = {}".format(len(english)))
print("Training Size = {}".format(trainingSize))
print("Test Size = {}".format(len(english) - trainingSize))
print("English Vocabulary Size = {}".format(englishVocabSize))
print("French Vocabulary Size = {}".format(frenchVocabsize))
print("Max Length of English Training Data = {}".format(englishMaxlength))
print("Max Length of French Training Data = {}".format(frenchMaxLength))


trainX = encodeSequences(trainEng, englishTokenizer, englishMaxlength)
trainY = encodeSequences(trainFr, frenchTokenizer, frenchMaxLength)
trainY = encodeOutput(trainY, frenchVocabsize)


testX = encodeSequences(testEng, englishTokenizer, englishMaxlength)
testY = encodeSequences(testFr, frenchTokenizer, frenchMaxLength)
trainY = encodeOutput(testY, frenchVocabsize)
