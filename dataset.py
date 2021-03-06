import pickle
from unidecode import unidecode
import numpy as np


def loadFile(filename):
    DataFile = open(filename, 'r')
    Data = DataFile.read()
    DataFile.close()
    return Data


def loadText(Text):
    eng = []
    fr = []
    for line in Text.split('\n'):
        if len(line) < 2:
            continue
        lang = line.split('\t')
        # print(lang)
        eng.append(lang[0])
        fr.append(lang[1])
    return np.array(eng), np.array(fr)


text = loadFile('dataset1/fra.txt')
english, french = loadText(text)


def clean(text):
    k = []
    for word in text:
        l = len(word)
        m = []
        for i in range(l):
            if word[i].isalpha() or word[i] == ' ':
                m.append(word[i].lower())
            else:
                continue
        k.append("".join(m))
    return np.array(k)


for i in range(len(french)):
    sequence = french[i]
    sequence = unidecode(sequence)
    french[i] = sequence


english = clean(english)
french = clean(french)


pickle.dump(english, open('english.pkl', 'wb'))
pickle.dump(french, open('french.pkl', 'wb'))
