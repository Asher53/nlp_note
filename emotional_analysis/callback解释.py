'''
回调函数Callbacks
'''

'''
回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。
通过传递回调函数列表到模型的.fit()中，即可在给定的训练阶段调用该函数集中的函数。
【Tips】虽然我们称之为回调“函数”，但事实上Keras的回调函数是一个类，回调函数只是习惯性称呼
'''

'''
回调函数以字典logs为参数，该字典包含了一系列与当前batch或epoch相关的信息。
目前，模型的.fit()中有下列参数会被记录到logs中：
在每个epoch的结尾处（on_epoch_end），logs将包含训练的正确率和误差，acc和loss，
如果指定了验证集，还会包含验证集正确率和误差val_acc)和val_loss，val_acc还额外需要在.compile中启用metrics=['accuracy']。
在每个batch的开始处（on_batch_begin）：logs包含size，即当前batch的样本数
在每个batch的结尾处（on_batch_end）：logs包含loss，若启用accuracy则还包含acc
'''