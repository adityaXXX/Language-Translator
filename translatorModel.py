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
from keras.callbacks import ModelCheckpoint


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
    encoder = tokenizer.texts_to_sequences(trainingData)
    encoder = pad_sequences(encoder, maxlen=maxlength, padding='pre')
    return encoder


def encodeOutput(testData, vocabSize):
    y = to_categorical(testData, num_classes = vocabSize)
    return y


def createModel(engVocab, frVocab, size, englishMaxlength, frenchMaxLength):
    model = Sequential()
    model.add(Embedding(input_dim = engVocab, output_dim = size, input_length = englishMaxlength, mask_zero = True))
    model.add(LSTM(units = size))
    model.add(RepeatVector(frenchMaxLength))
    model.add(LSTM(units = size, return_sequences = True))
    model.add(TimeDistributed(Dense(frenchVocabsize, activation = 'softmax')))
    print(model.summary())
    return model


def DataGenerator(trainingDataEnglish, trainingDataFrench):
    while True:
        l = len(trainingDataFrench)
        for i in range(l):
            arr1 = np.array(trainingDataEnglish[i])
            arr2 = np.array(trainingDataFrench[i])
            arr1 = np.expand_dims(arr1, axis = 0)
            arr2 = np.expand_dims(arr2, axis = 0)
            yield(arr1, arr2)


english = loadLanguageFile('english.pkl')
french = loadLanguageFile('french.pkl')


samples = 8000
trainingSize = 6000
trainEng = english[:trainingSize]
trainFr = french[:trainingSize]
testEng = english[trainingSize:samples]
testFr = french[trainingSize:samples]
englishTokenizer = createTokenizer(trainEng)
frenchTokenizer = createTokenizer(trainFr)
englishVocabSize = len(englishTokenizer.word_index) + 1
frenchVocabsize = len(frenchTokenizer.word_index) + 1
englishMaxlength = maxLength(trainEng)
frenchMaxLength = maxLength(trainFr)


print("Dataset size = {}".format(len(english)))
print("Training Size = {}".format(trainingSize))
print("Test Size = {}".format(samples - trainingSize))
print("English Vocabulary Size = {}".format(englishVocabSize))
print("French Vocabulary Size = {}".format(frenchVocabsize))
print("Max Length of English Training Data = {}".format(englishMaxlength))
print("Max Length of French Training Data = {}".format(frenchMaxLength))


trainX = encodeSequences(trainEng, englishTokenizer, englishMaxlength)
trainY = encodeSequences(trainFr, frenchTokenizer, frenchMaxLength)
trainY = encodeOutput(trainY, frenchVocabsize)


testX = encodeSequences(testEng, englishTokenizer, englishMaxlength)
testY = encodeSequences(testFr, frenchTokenizer, frenchMaxLength)
testY = encodeOutput(testY, frenchVocabsize)


epochs = 20

# print(trainX.shape)
# print(trainY.shape)
model = createModel(engVocab = englishVocabSize, frVocab = frenchVocabsize, size = 256, englishMaxlength = englishMaxlength, frenchMaxLength = frenchMaxLength)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, mode = 'min')
model.fit(trainX, trainY, epochs = epochs, callbacks = [checkpoint], validation_data = (testX, testY))


# steps = len(trainX)
# generator = DataGenerator(trainX, trainY)
# e, f = next(generator)
# print(e)
# print(f.shape)
# model.fit_generator(generator, epochs = epochs, steps_per_epoch = steps, validation_data = (testX, testY))
# model.save('Model.h5')
