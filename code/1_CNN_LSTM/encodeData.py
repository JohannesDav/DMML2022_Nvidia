import pandas as pd
from nltk import RegexpTokenizer
import io
import numpy as np
import unidecode
import spacy
import tqdm


def load_vec(emb_path, nmax=200000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

embeddings, id2word, word2id = load_vec('cc.fr.300.vec')
levelsdf = pd.read_csv('FLELex_TreeTagger.csv', sep='	')
nlp = spacy.load('fr_dep_news_trf')

def embTranform(word):
    if word in word2id:
        return 1, embeddings[word2id[word]]
    else:
        # print("Word not found in embeddings: ", word)
        return 0, np.zeros(300)

def levelTransform(word):
    if word in levelsdf['word'].values:
        levelFrequencies = levelsdf.loc[levelsdf['word'] == word.lower(), 'freq_A1':'freq_C2'].values[0]
        totFreq = levelsdf.loc[levelsdf['word'] == word.lower(), 'freq_total'].values[0]
        if (levelFrequencies.max() - levelFrequencies.min()) > 0:
            levelFrequencies2 = (levelFrequencies - levelFrequencies.min()) / (levelFrequencies.max() - levelFrequencies.min())
        else:
            return 0, np.zeros(7)
        # map total frequencies to a log distribution between 0 and 1
        maxFreq = 85513.6877
        minNormFreq = -17.13278489966132
        maxNormFreq = 0
        totFreq = totFreq / maxFreq
        totFreq = np.log(totFreq)
        totFreq = (totFreq - minNormFreq) / (maxNormFreq - minNormFreq)
        allFreq = levelFrequencies2
        allFreq = np.append(allFreq, totFreq)
        return 1, allFreq
    else:
        # print("Word not found in levels: ", word)
        return 0, np.zeros(7)

def tryTransform(transform, word):
    # progessively denature the word hoping for a match
    success, result = transform(word)
    if not success:
        #lowercase
        wordMod = word.lower()
        success, result = transform(wordMod)
    if not success:
        #remove symbols
        symbols = set(r"""`~!@#$%^&*()_-+={[}}|\:;"'<,>.?/""")
        wordMod = ''.join(ch for ch in word if ch not in symbols)
        success, result = transform(wordMod)
    if not success:
        #lowercase and remove symbols
        wordMod = wordMod.lower()
        success, result = transform(wordMod)
    if not success:
        #lowercase and remove symbols and accents
        wordMod = unidecode.unidecode(wordMod)
        success, result = transform(wordMod)
    if not success:
        # lemmatize
        doc = nlp(word)
        wordMod = doc[0].lemma_
        success, result = transform(wordMod)
        
    return success, result

def encodeSentence(sentence):
    toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
    words = toknizer.tokenize(sentence)
    SentenceEncoding = []
    for word in words:
        success1, wordEmb = tryTransform(embTranform, word)
        success2, wordLevel = tryTransform(levelTransform, word)
        if success1: # level is missing for too many tokens
            SentenceEncoding.append(np.concatenate((wordLevel, wordEmb), axis=0))
    # keep only the first 75 words
    SentenceEncoding = SentenceEncoding[:75]
    # pad with np.zeros(313) if less than 75 words
    if len(SentenceEncoding) < 75:
        SentenceEncoding = SentenceEncoding + [np.zeros(307)] * (75 - len(SentenceEncoding))
    return np.array(SentenceEncoding)


traindf = pd.read_csv('training_data.csv', sep=',')
testdf = pd.read_csv('unlabelled_test_data.csv', sep=',')

trainSentences = traindf['sentence'].tolist()
testSentences = testdf['sentence'].tolist()

trainEncodings = []
for sentence in tqdm.tqdm(trainSentences):
    SentenceEncoding = encodeSentence(sentence)
    trainEncodings.append(SentenceEncoding)
trainEncodings = np.array(trainEncodings)
print(trainEncodings.shape)
np.save('trainEncodings.npy', trainEncodings)

testEncodings = []
for sentence in tqdm.tqdm(testSentences):
    SentenceEncoding = encodeSentence(sentence)
    testEncodings.append(SentenceEncoding)
testEncodings = np.array(testEncodings)
print(testEncodings.shape)
np.save('testEncodings.npy', testEncodings)