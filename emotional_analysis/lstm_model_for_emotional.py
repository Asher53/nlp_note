'''
数据集为UCI公开数据集 Roman Urdu DataSet
这些数据，均为文本数据。原始数据的文本，对应三类情感标签：Positive, Negative, Netural。
本练习赛，移除了标签为Netural的数据样例。因此，练习赛中，所有数据样例的标签为Positive和Negative。
本练习赛的任务是「分类」。「分类目标」是用训练好的模型，对测试集中的文本情感进行预测，判断其情感为「Negative」或者「Positive」
'''

'''
分为 
1 数据清洗
2 keras.preprocessing.text.Tokenizer and pad_sequences
3 embedding
4 lstm
5 train and predict
'''

import numpy as np
import pandas as pd
import warnings;

warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

df_train = pd.read_csv('train.csv', lineterminator='\n')
df_test = pd.read_csv('test.csv', lineterminator='\n')

df_train['label'] = df_train['label'].map({'Negative': 0, 'Positive': 1})

numpy_array = df_train.as_matrix()
numpy_array_test = df_test.as_matrix()


# two commom ways to clean data
def cleaner(word):
    word = re.sub(r'\#\.', '', word)
    word = re.sub(r'\n', '', word)
    word = re.sub(r',', '', word)
    word = re.sub(r'\-', ' ', word)
    word = re.sub(r'\.', '', word)
    word = re.sub(r'\\', ' ', word)
    word = re.sub(r'\\x\.+', '', word)
    word = re.sub(r'\d', '', word)
    word = re.sub(r'^_.', '', word)
    word = re.sub(r'_', ' ', word)
    word = re.sub(r'^ ', '', word)
    word = re.sub(r' $', '', word)
    word = re.sub(r'\?', '', word)

    return word.lower()


def hashing(word):
    word = re.sub(r'ain$', r'ein', word)
    word = re.sub(r'ai', r'ae', word)
    word = re.sub(r'ay$', r'e', word)
    word = re.sub(r'ey$', r'e', word)
    word = re.sub(r'ie$', r'y', word)
    word = re.sub(r'^es', r'is', word)
    word = re.sub(r'a+', r'a', word)
    word = re.sub(r'j+', r'j', word)
    word = re.sub(r'd+', r'd', word)
    word = re.sub(r'u', r'o', word)
    word = re.sub(r'o+', r'o', word)
    word = re.sub(r'ee+', r'i', word)
    if not re.match(r'ar', word):
        word = re.sub(r'ar', r'r', word)
    word = re.sub(r'iy+', r'i', word)
    word = re.sub(r'ih+', r'eh', word)
    word = re.sub(r's+', r's', word)
    if re.search(r'[rst]y', 'word') and word[-1] != 'y':
        word = re.sub(r'y', r'i', word)
    if re.search(r'[bcdefghijklmnopqrtuvwxyz]i', word):
        word = re.sub(r'i$', r'y', word)
    if re.search(r'[acefghijlmnoqrstuvwxyz]h', word):
        word = re.sub(r'h', '', word)
    word = re.sub(r'k', r'q', word)
    return word


def array_cleaner(array):
    # X = array
    X = []
    for sentence in array:
        clean_sentence = ''
        words = sentence.split(' ')
        for word in words:
            clean_sentence = clean_sentence + ' ' + cleaner(word)
        X.append(clean_sentence)
    return X


X_test = numpy_array_test[:, 1]
X_train = numpy_array[:, 1]
# Clean X here
X_train = array_cleaner(X_train)
X_test = array_cleaner(X_test)
y_train = numpy_array[:, 2]

y_train = np.array(y_train)
y_train = y_train.astype('int8')

X_all = X_train + X_test  # Combine both to fit the tokenizer.
lentrain = len(X_train)

tokenizer = Tokenizer(
    nb_words=2000,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=' ')
tokenizer.fit_on_texts(X_all)

X = tokenizer.texts_to_sequences(X_all)

X = pad_sequences(X)

embed_dim = 128
lstm_out = 256
batch_size = 32

model = Sequential()
model.add(Embedding(2000, embed_dim, input_length=X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2, return_sequences=True))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics = ['accuracy'])
print(model.summary())

X_train = X[:lentrain]  # Separate back into training and test sets.
X_test = X[lentrain:]
y_binary = to_categorical(y_train)

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch + 1, score))


#             print(y_pred[:6])
x_train5, y_train5, x_label5, y_label5 = train_test_split(X_train, y_binary, train_size=0.8, random_state=234)
RocAuc = RocAucEvaluation(validation_data=(y_train5, y_label5), interval=1)

hist = model.fit(x_train5, x_label5, batch_size=batch_size, epochs=1, validation_data=(y_train5, y_label5),
                 callbacks=[RocAuc], verbose=2)

y_lstm = model.predict_proba(X_test, batch_size=batch_size)[:, 1]

lstm_output = pd.DataFrame(data={"ID": df_test["ID"], "Pred": y_lstm})
lstm_output.to_csv('lstm_new.csv', index=False, quoting=3)
