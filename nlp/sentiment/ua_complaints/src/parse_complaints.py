import numpy as np
import math
import operator # for sorting dict (by value or key)
import os
import nltk
from BeautifulSoup import BeautifulSoup
#nltk.download('stopwords')
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


DATA_DIR = "../dat/html/"

def read_file(path):
    f = open(path, 'r')
    contents = f.read()
    f.close()
    return contents

def parse_complaint(html):
    soup = BeautifulSoup(''.join(html))
    complaints = []
    ratings = []
    for hit in soup.findAll(attrs={'class' : 'review-body__text'}):
        try:
            complaints.append(hit.contents[0].text.encode('utf-8').replace("\xe2\x80\x99","'").replace("\xe2\x80\x9c","'").replace("\xe2\x80\x9d","'").lower())
        except:
            complaints.append(hit.contents[0].encode('utf-8').replace("\xe2\x80\x99","'").replace("\xe2\x80\x9c","'").replace("\xe2\x80\x9d","'").lower())

    for hit in soup.findAll(attrs={'itemprop' : 'ratingValue'}):
        ratings.append(hit)

    ratings = map(lambda x: int(x['content']), ratings[1:])
    assert(len(complaints) == len(ratings))

    return zip(complaints, ratings)

### Test ###
def read_all_complains(path):
    out = []
    success = 0
    failure = 0
    for filename in os.listdir(path):
        html = read_file(path + filename)
        try:
            ua_complaints = parse_complaint(html)
            print "Successfully parsed: " + path + filename
            success += 1
        except Exception:
            print "couldn't parse: " + path + filename
            out.extend(ua_complaints)
            failure += 1
    print 'failure: ', failure
    return out

#html = read_file(DATA_DIR + '70.html')
#ua_complaints = parse_complaint(html) # no <p> tag. unsuccessful
ua_complaints = read_all_complains(DATA_DIR)

### Tokenizer to tokenize (separate) a string of text
### into individual tokens (words). This is not necessarily 
### done by spaces, but also by apostrophes. 
### e.g. "I don't like you" -> ["I", "do", "n't", "like", "you"]
tokenizer = nltk.tokenize.TreebankWordTokenizer()

### Normalization
# - Stemmers find the stems of words by word chopping. cats -> cat
# - Lemmatizer finds the root of words by morphing. wolves -> wolf
stemmer = nltk.stem.PorterStemmer() 
#stemmer = nltk.stem.WordNetLemmatizer()

### Stop words ###
# Words that are common. Like "a, and, the, ..."
stopwords = set(nltk.corpus.stopwords.words('english'))

def get_ngrams(text, n=2):
    try:
        #return list( nltk.trigrams(x.split()) )
        text = " ".join(filter(lambda w: not w in stopwords, text.split()))
        tokens = tokenizer.tokenize(text)
        x = [stemmer.stem(token) for token in tokens]
        #x = " ".join(stemmer.lemmatize(token) for token in tokens)
        #return list( nltk.trigrams(x) )
        return ["_".join(ngram) for ngram in list( nltk.ngrams(x, n) )]
    except:
        return ""

all_ngrams = map(lambda (complaint,_) : get_ngrams(complaint, 1), ua_complaints)

ngram_freq = dict()
for text in all_ngrams:
    for ngram in text:
        ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1

common_words = filter(lambda ngram: 25 < ngram_freq[ngram] < 200, ngram_freq)
common_words = [ (w, ngram_freq[w]) for w in common_words]
common_words = sorted(common_words, key=operator.itemgetter(1), reverse=True)
for (w,c) in common_words: print w,c

def word2vec(complaint, common_words):
    try:
        text,y = complaint
        text = " ".join(filter(lambda w: not w in stopwords, text.split()))
        tokens = tokenizer.tokenize(text)
        stems = [stemmer.stem(token) for token in tokens]
        x = [ (w in stems) * 1 for w in common_words]
        return (y, x)
    except:
        return ()


data = map(lambda c: word2vec(c, dict(common_words).keys()), ua_complaints)
data = filter(lambda d: d != (), data)
N = len(data)
K = len(common_words)
X = np.zeros((N,K+1))
y = np.zeros(N)
for i in range(N):
    yi, xi = data[i]
    y[i] = yi > 2
    X[i, 1:] = xi

n_train = int(N * .7)
idx = np.random.permutation(N)
train_idx = idx[:n_train]
test_idx = idx[n_train:]

### Not a good idea ###

#logreg = LogisticRegression()
#mod = logreg.fit(X[train_idx,:], y[train_idx])
#mod.predict(X[test_idx])
#y[test_idx]
#np.mean(y[test_idx] == mod.predict(X[test_idx, :]))



### TODO: do this for all files, 1.html, 2.html, ..., 70.html
### TODO: bi-grams, tri-grams. Logistic regression. ratings ~ complaints

# MAIN
#parse_complaints('../dat/html/', '../dat/complaints/')
