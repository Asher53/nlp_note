'''
GloVe模型
'''

'''
共现矩阵
在整个语料库中，单词i和单词j共同出现在一个窗口中的次数
'''

'''
Cbow/Skip-Gram 是一个local context window的方法，比如使用NS来训练，缺乏了整体的词和词的关系，负样本采用sample的方式会缺失词的关系信息。
另外，直接训练Skip-Gram类型的算法，很容易使得高曝光词汇得到过多的权重
Global Vector融合了矩阵分解Latent Semantic Analysis (LSA)的全局统计信息和local context window优势。
融入全局的先验统计信息，可以加快模型的训练速度，又可以控制词的相对权重。
我的理解是skip-gram、CBOW每次都是用一个窗口中的信息更新出词向量，但是Glove则是用了全局的信息（共现矩阵），也就是多个窗口进行更新
'''