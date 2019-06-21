'''
dropout
'''

'''
一般，我们会将dropout理解为“一种低成本的集成策略”，这是对的，具体过程可以大概这样理解：
经过上述置零操作后，我们可以认为零的部分是被丢弃的，丢失了一部分信息。因而，逼着模型用剩下的信息区拟合目标。
然而每次dropout是随机的。我们就不能侧重于某些节点，所以总的来说就是—每次逼着模型用少量的特征学习，
每次被学习的特征又不同，那么就是说，每个特征都应该对模型的预测有所贡献（而不是侧重于部分特征，导致过拟合）。
'''

'''
首先，我们初始化一个1x7x5的三维张量，如下所示：
从结果中，我们可以看到普通的dropout随机地将部分元素置零，并且是无规律的，也就是说下次可能是另外一部分元素置零。
'''

# encoding:utf-8
import numpy as np
import keras.backend as K

ary = np.arange(35).reshape((1, 7, 5))
inputs = K.variable(ary)
print(ary)

'''
可以看到，与普通的dropout不同的是，SpatialDropout1D随机地将某块区域(列或者行)全部置零。
'''
# 普通的drop_out
dropout_1 = K.eval(K.dropout(inputs, level=0.5))
print(dropout_1)

# SpatialDropout1D
input_shape = K.shape(inputs)
noise_shape = (input_shape[0], 1, input_shape[2])
dropout_2 = K.eval(K.dropout(inputs, 0.5, noise_shape))
print(dropout_2)
