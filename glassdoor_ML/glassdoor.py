import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from pyspark.ml.classification import RandomForestClassifier
import sys
import util
import nn

### TODO: Spark?

### Read Data
DATA_DIR = 'dat/jobs.csv'
dat = pd.read_csv(DATA_DIR)
#dat = spark.read.csv(DATA_DIR, header=True)
colnames = list(dat)
dat = dat.as_matrix()
num_obs = len(dat)
#sys.getsizeof(dat)

unique_users = np.array(list(set(dat[:, -2]))).flatten()
unique_mgoc = np.array(list(set(dat[:, -1]))).flatten()

def vec_user(x):
    v = np.zeros(len(unique_users))
    i = np.argwhere(unique_users == x)[0,0]
    v[i] = 1
    return v

def vec_mgoc(x):
    v = np.zeros(len(unique_mgoc))
    i = np.argwhere(unique_mgoc == x)[0,0]
    v[i] = 1
    return v

#x = vec_user(dat_train['X'][0,-2])
#x = vec_mgoc(dat_train['X'][0,-1])

### Format Data -> X, y
def toXY(data):
    X = data[:,:7]

    #X = data[:,[0,1,2,3,4,5,6,9,10]]
    Z = np.array(map(vec_mgoc, data[:,-1]))
    X = np.concatenate((X,Z),1)
    #X[:,-2] = map(vec_user, X[:,-2])

    y = data[:,7]
    return {'X': X, 'y': y}

def upsampleClass(data, c):
    """
    c: the class to upsample
    """
    N = len(data['y'])
    idx_minority = np.argwhere(data['y'] == c).flatten()
    num_majority = N - len(idx_minority)
    idx_minority_new = np.random.choice(idx_minority, num_majority)
    idx_majority = np.argwhere(data['y'] != c).flatten()
    idx_new = np.concatenate( (idx_minority_new, idx_majority) )
    X_new = data['X'][idx_new]
    return {'X': X_new + np.random.normal(0, .1, X_new.shape),
            'y': data['y'][idx_new]}


### Look at unique dates
#all_dates = sorted(set(dat['search_date_pacific']))

### Split data to train and test set
search_date = dat[:,np.argwhere(np.array(colnames) == 'search_date_pacific')].flatten()
test_date = '2018-01-27'
isTestIdx = search_date == test_date
test_idx = np.argwhere(search_date == test_date).flatten()
train_and_val_idx = np.argwhere(search_date != test_date).flatten()
np.random.shuffle(train_and_val_idx)
validation_prop = .3
validation_size = int(validation_prop * num_obs)

dat_test = toXY(dat[test_idx,:])
dat_val = toXY(dat[train_and_val_idx[:validation_size], :])
dat_train = toXY(dat[train_and_val_idx[validation_size:], :])



### Model
print "Training Model..."
#rf = RandomForestClassifier(max_depth=2, n_estimators=50)
rf = RandomForestClassifier()
#logreg = LogisticRegression(C=.01, penalty='l1')

def fit_model(data):
    return rf.fit(data['X'], data['y'])
    #return logreg.fit(data['X'], data['y'])


### Logistic Regression
N = 100000
u_train = upsampleClass(dat_train, 1)
u_train['X'] = u_train['X'][:N]
u_train['y'] = u_train['y'][:N]
mod = fit_model(upsampleClass(u_train, 1))
pred_val = mod.predict(dat_val['X'])
conf = util.confusion(pred_val, dat_val['y'], numClasses=2)
print conf
tn = conf[0,0] / np.sum(dat_val['y'] == 0)
tp = conf[1,1] / np.sum(dat_val['y'] == 1)
tn
tp

### Neural Network
#u_train = upsampleClass(dat_train, 1)
#u_val = upsampleClass(dat_val, 1)
#mod = nn.nn_model(u_train['X'], nn.one_hot(u_train['y'].astype(int),2),
#                  u_val['X'], nn.one_hot(u_val['y'].astype(int),2), 
#                  hidden_layer_size=4, 
#                  lam=1.5, learning_rate=.001,
#                  num_epochs=20, mini_batch_size=10000)
#
#pred = nn.predict(dat_test['X'], mod['parameters'])
#conf = util.confusion(pred, dat_test['y'], numClasses=2)
#
#print conf
#tn = conf[0,0] / np.sum(dat_test['y'] == 0)
#tp = conf[1,1] / np.sum(dat_test['y'] == 1)
#tn
#tp

### Try Next:
# model the user
