# -*- coding: utf-8 -*-
"""
Created on Sat May  8 23:52:10 2021

@author: MaYiming
"""
from bayes_classifier import Bayes_classifier

#训练集
postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
classVec = [0,1,0,1,0,1]    #1是侮辱性质，0是非侮辱性质的

#初始化模型
classifier = Bayes_classifier(postingList,classVec)

#测试集
testdata = [['love', 'my', 'dalmation'],['stupid', 'garbage']]
#测试
for index,each in enumerate(testdata):
    #调用构建的classifier的classify函数进行预测
    test = classifier.classify(each)
    print("第%d个测试数据的结果为%d"%(index+1,test))