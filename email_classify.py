# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:38:30 2021

@author: MaYiming
"""
#导入库
import os
from bayes_classifier import Bayes_classifier
import numpy as np
import matplotlib.pyplot as plt
#定义参数，包括两个路径和路径下的文件数量
params = {}
params["ham_path"] = 'email\\ham\\'
params["spam_path"] = 'email\\spam\\'


#获得数据
def Getdata(params):
    dataset = []
    label = []
    #利用for_loop得到每个垃圾邮件数据
    for index in range(params["spam_num"]):
        #文件路径
        path = os.path.join(params["spam_path"],str(index+1)+'.txt')
        #得到split之后的列表
        lis = GetAllAlpha(path)
        #正类
        label.append(1)
        dataset.append(lis)
    #利用for_loop得到每个正常邮件数据
    for index in range(params["ham_num"]):
        path = os.path.join(params["ham_path"],str(index+1)+'.txt')
        lis = GetAllAlpha(path)
        #负类
        label.append(0)
        dataset.append(lis)
    return dataset,label

def GetAllAlpha(path):
    #定义两个空字符
    lis_line = ""
    lis = ""
    #打开文件
    with open(path) as p:
        for line in p:
            #去除每行的换行
            line = line.strip("\n")
            lis_line += line
            #加一个空格，用以split(" ")
            lis_line = lis_line + ' '
    #isblank用于不要重复加“ ”
    isblank = 0
    for li in lis_line:
        #只有字母和数字才能进入
        if (li.isalnum()):
            lis += li
            isblank = 0
        #如果已经加了空格就不会进入这
        elif isblank == 0:
            lis += " "
            isblank = 1
    #split
    return lis.strip(" ").split(" ")

accuracy = []
for i in range(1,24):
    params["num"] = i
    params["ham_num"] = params["num"]
    params["spam_num"] = params["num"]
    #获得数据
    dataset,label = Getdata(params)
    #得到我们的路径
    classifier = Bayes_classifier(dataset,label)
    #分别找5个ham和5个spam来预测
    test_tuple = []
    for index in range(25-params["num"]):
        #测试数据路径
        test_path = os.path.join("email//ham//",str(params["num"]+index+1)+".txt")
        #加入对应标签
        test_tuple.append((test_path,0))
    
    for index in range(25-params["num"]):
        #测试数据路径
        test_path = os.path.join("email//spam//",str(params["num"]+index+1)+".txt")
        #加入对应标签
        test_tuple.append((test_path,1))
    #转化为np数组
    test_tuple = np.array(test_tuple)
    #洗牌
    np.random.shuffle(test_tuple)
    #测试
    error = 0
    for i in range(2*(25-params["num"])):
        #得到分词
        test_lis = GetAllAlpha(test_tuple[i][0])
        #预测
        predict = classifier.classify(test_lis)
        if int(test_tuple[i][1]) != int(predict):
            error += 1
        #print("该数据标签为{}，预测结果为{}".format(test_tuple[i][1],predict))
    accuracy.append(error/(2*(25-params["num"])))
plt.figure()
plt.xlabel("Trainning Number")
plt.ylabel("Error Rate")
plt.plot(accuracy)