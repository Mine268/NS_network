#!/usr/bin/env python
# coding=utf-8

'''
这一个类是神经网络中的单层类，引用了matrix类，以下是方法与成员说明
成员:
    __roll: 储存这一层的长度
    __weights: 从上一层到这一层计算所需的权重矩阵
    __bias: 这一层加上的偏置矩阵
    __data: 这一层的神经元值，经过了压缩函数的计算
    __z_mat: 这一层的输入值
方法:
    __init__(self, weight_mat, bias_mat): 构造函数，输入权重矩阵和偏置矩阵（向量）
    getResult(self, prev_layer): 通过prev_layer（上一层的输出）计算出这一层的结果
    @__copy(mat): 根据引用mat生成mat的矩阵的一个深拷贝，并返回
    @squeeze(val): 挤压函数
    calcData(self, prev_layer): 通过上一层计算这一层
    ...unfinish
'''

import matrix

class layer(object):
    __roll = -1 # 这一层的长度
    # 以下成员均为matrix类型
    __weights = -1 # 计算到这一层所需要的权重
    __bias = -1 # 偏置
    __data = -1 # 这一层的神经元
    __z_mat = -1 # 这里用来暂存该层的输入，用于方向传播

    # 构造函数，需要输入权重矩阵和偏置矩阵，类型为matrix
    def __init__(self, weight_mat, bias_mat):
        # 读如权重矩阵和偏置矩阵的长宽，bias默认为1
        weight_roll, weight_column = weight_mat.getSize()[0], weight_mat.getSize()[1]
        bias_roll = bias_mat.getSize()[0]
        
        # 检查没有矩阵大小上的错误
        if (weight_roll != bias_roll):
            raise Exception(print(f"权重矩阵和偏置矩阵不相容。（{weight_roll}x{weight_column} {bias_roll}x{1}）"))
        
        # 初始化权重，神经元和偏置并置为零，依据权重和偏置的数据初始化神经元的数量
        self.__weights = matrix.matrix.zero_mat(weight_roll, weight_column)
        self.__data = matrix.matrix.zero_mat(weight_roll, 1)
        self.__z_mat = matrix.matrix.zero_mat(weight_roll, 1)
        self.__bias = matrix.matrix.zero_mat(weight_roll, 1)
        self.__roll = weight_roll

        # 注意深拷贝，注意matrix类中下标从1开始计算
        # 将weight_mat中的数据拷贝入__weights中
        for i in range(0, weight_roll):
            for j in range(0, weight_column):
                self.__weights[i][j] = weight_mat.readAt(i + 1, j + 1)

        # 将bias_mat中的数据拷贝到__bias中
        for i in range(0, bias_roll):
            self.__bias[i] = bias_mat.readAt(i + 1, 1)

    # 获取这一层的计算结果
    def getResult(self):
        return self.__data # 注意，这里返回的是引用，因为进行计算的时候不需要更改本行的数据

    # 这是一个返回深拷贝的方法，类型为matrix，备用
    @staticmethod
    def __copy(mat):
        size = mat.getSize()
        res = matrix.matrix.zero_mat(size[0], size[1])
        for i in range(1, size[0] + 1):
            for j in range(1, size[1] + 1):
                res.setAt(i, j, mat.readAt(i, j))
        return res

    # 挤压函数
    @staticmethod
    def squeeze(val):
        if (val <= 0):
            return 0
        else:
            return val

    # 通过输入进行计算
    def calcData(self, prev_layer):
        self.__data = matrix.matrix.product(self.__weights, prev_layer)
        self.__data += matrix.matrix.plus(self.__data, self.__bias)

        # 配置__z_mat矩阵，以便后期进行方向传播
        self.__z_mat = matrix.__copy(self.__data)

        # 通过挤压函数挤压每个值
        for i in range(1, self.__roll + 1):
            self.__data.setAt(i, 1, squeeze(self.__data.readAt(i, 1)))
