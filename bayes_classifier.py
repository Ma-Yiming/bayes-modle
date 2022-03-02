# -*- coding: utf-8 -*-
"""
Created on Sat May  8 08:41:44 2021

@author: MaYiming
"""
import numpy as np

class Bayes_classifier:
    #我们的目标是知道特征之后预测类
    #利用贝叶斯公示，我们可以分别预测0类和1类的概率，谁大就将谁作为输出
    #因此需要我们通过数据库算出p(特征)，p(类别),p(特征|类别)
    #但是贝叶斯预测时p(特征)分母都是相同的，所以这个可以不算
    def __init__(self,dataset,label):
        #初始化数据集和标签
        self.dataset = dataset
        self.label = label
        #创建词向量，这里的每个词都是一个特征，方法有点笨，但是可行
        self.P_categoryset = []
        #词向量的长度初始化为0
        self.featureNum = 0
        #python中的set可创建内容不重复的数据结构
        self.vocabSet = set([])
        #类已知时特征的条件概率集合
        self.P_feature_categoryset = []
        #计算数据库给定的情况下的类别概率p(类别)和类已知的特征条件概率p(特征|类别)
        self.LoadAndInit(self.dataset,self.label)
        
    def LoadAndInit(self,dataset,label):
        #计算正类和负类的概率，即p(类别)
        self.P_categoryset = self.P_category(self.label)
        
        #初始化特征向量
        for sentence in dataset:
            self.vocabSet = self.vocabSet | set(sentence)
        self.vocabSet = list(self.vocabSet)
        
        #计算类已知时对应特征概率向量，即p(特征|类别)
        self.P_feature_categoryset = self.P_feature_category(dataset,label,self.vocabSet)
        
        #至此我们两个所需的概率都算出来了

    def P_feature_category(self,dataset,label,vocabSet):
        #计算贝叶斯概率表
        self.featureNum = len(vocabSet)
        featureNum = self.featureNum
        featureMatrix = [[0]*featureNum for i in range(len(dataset))]
        #计算每句话的特征向量
        for i,sentence in enumerate(dataset):
            for word in sentence:
                if word in vocabSet:
                    featureMatrix[i][vocabSet.index(word)] = 1
                    
        #拉普拉斯修正，但是1有点大
        p_feature_1 = np.ones(featureNum)
        p_feature_0 = np.ones(featureNum)
        #出现总数也拉普拉斯修正，加上特征数
        p_feature_1_all = float(featureNum)
        p_feature_0_all = float(featureNum)
        #计算正类和负类每个向量对应的概率
        for index,each in enumerate(label):
            if each == 1:
                p_feature_1 += featureMatrix[index]
                p_feature_1_all += sum(featureMatrix[index])
            else:
                p_feature_0 += featureMatrix[index]
                p_feature_0_all += sum(featureMatrix[index])
        #得出概率
        p_feature_1 = p_feature_1/p_feature_1_all
        p_feature_0 = p_feature_0/p_feature_0_all
        #这里计算出来了正类和负类每个特征出现的概率，之间是相互独立的
        #所以两个特征同时发生的概率是分别的相乘
        return [p_feature_0,p_feature_1]
    def P_category(self,label):
        total_len = len(label)
        #拉普拉斯修正
        total_zero = 0.1
        total_one = 0.1
        
        for each in label:
            if each == 0:
                total_zero += 1
            else:
                total_one += 1
        #返回正类和负类的先验概率
        return [total_zero/total_len,total_one/total_len]
    def P_test_feature(self, data):
        test_featureVec = [0]*self.featureNum
        #计算测试句子的特征向量
        for word in data:
            if word in self.vocabSet:
                test_featureVec[self.vocabSet.index(word)] = 1
        #计算这个测试句子的p(类别|特征)
        pFeature0 = np.exp(sum(test_featureVec*np.log(self.P_feature_categoryset[0])))
        pFeature1 = np.exp(sum(test_featureVec*np.log(self.P_feature_categoryset[1])))
        #返回
        return [pFeature0,pFeature1]
    def classify(self,data):
        #利用贝叶斯分别预测比较大小得出类别
        p1 = self.P_test_feature(data)[1]*self.P_categoryset[1]
        p0 = self.P_test_feature(data)[0]*self.P_categoryset[0]
        if p1 > p0:
            return 1
        else:
            return 0