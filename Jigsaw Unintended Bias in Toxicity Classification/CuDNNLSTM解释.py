'''
CuDNNLSTM与LSTM都是keras.layers里的实现lstm的单元。那么二者有什么区别呢？CuDNNLSTM肯定是只能用在GPU下，但是LSTM也是可以用在GPU下的啊。
所以问题是：在你拥有GPU资源的情况下（默认拥有）,我应该选用哪一种呢？
答案是CuDNNLSTM！
'''

'''
1.原因：
In my case, training a model with LSTM took 10mins 30seconds. Simply switching the call from LSTM() to CuDNNLSTM() took less than a minute.

I also noticed that switching to CuDNNLSTM() speeds up model.evaluate() and model.predict() substantially as well.
同样的数据集，LSTM()耗时10分半，CuDNNLSTM() 耗时1分钟。。。同时预测与评估也会变快很多！！！

2.原理：
CuDNN:Fast LSTM implementation backed by CuDNN. Can only be run on GPU, with the TensorFlow backend.

3.小不足：
CuDNNLSTM is faster (it uses the GPU support) but it has less options than LSTM (dropout for example)

'''