'''
LSTM要求的是每个string都是一个(num of word , length of word vector)

batch需要自定义
'''

from keras.layers import LSTM
from keras.models import Sequential

# time_step = 13
# featrue = 5
# hidenfeatrue = 10

time_step = 20
featrue = 200
hidenfeatrue = 128
batch = 64

model = Sequential()
model.add(LSTM(hidenfeatrue, batch_size=batch, input_shape=(time_step, featrue)))
model = Sequential()
model.summary()


'''
参数总数计算公式 
[(200+128)*128+128]*4
'''
'''
某时刻输出
[64,(200+128)] * [(200+128), (4*128)],  4代表4个神经元
output  [64,128]
'''