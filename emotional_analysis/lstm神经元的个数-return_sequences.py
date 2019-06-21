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
'''
一个三维向量？
很好理解，在初始版lstm中，我们每个时刻都会输出一个（32，256），而句子长度都是固定的219，
所以219个时刻一共会输出219个（32，256），而没有加return_sequences=True参数之前，他只会保留最后一个时刻的输出，
所以是（32，256），但是加了return_sequences=True参数之后，每个时刻的输出都会保留，那么输出就是219个（32，256），
也就是（32，219，256）。但是每个时刻的输出不变，所以参数不变。
同时，保留多个时刻的输出，经过我的验证，效果是有的。
'''
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 219, 128)          256000    
_________________________________________________________________
lstm_1 (LSTM)                (None, 219, 256)          394240    
_________________________________________________________________
flatten_1 (Flatten)          (None, 56064)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 112130    
=================================================================
Total params: 762,370
Trainable params: 762,370
Non-trainable params: 0
_________________________________________________________________
'''
embed_dim = 128
hidenfeatrue = 256
batch_size = 32

model = Sequential()
model.add(Embedding(2000, embed_dim, input_length=219, dropout=0.2))
model.add(LSTM(hidenfeatrue, dropout_U=0.2, dropout_W=0.2, return_sequences=True))
model.add(Flatten())  # 直接压扁 219 * 256 = 56064
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss = 'binary_crossentropy',optimizer='adam',metrics = ['accuracy'])
print(model.summary())
