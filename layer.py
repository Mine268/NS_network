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
    getLength(self): 返回这一层的层数
    squeeze_[](val): 挤压函数
    d_squeeze_[](val): 挤压函数的导数
    getWeight_mat(self): 获取权重矩阵
    getData_mat(self): 获取数据向量
    ## 以下五个函数由外部调用
    atZ(self, rol): 获得第rol层的输入
    atBias(self, rol): 获得第rol层的偏置
    atData(self, rol): 获得第rol层的输出
    atDelta(self, rol): 获得第rol层的delta
    atWeight(self, rol, col): 获得第rol,col处的权重
'''

import matrix
import math

class layer(object):
    __roll = -1 # 这一层的长度
    # 以下成员均为matrix类型
    __weights = -1 # 计算到这一层所需要的权重
    __bias = -1 # 偏置
    __data = -1 # 这一层的神经元
    __z_mat = -1 # 这里用来暂存该层的输入，用于方向传播
    __delta = -1 # 这一层是中间变量的层，bp

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
        self.__delta = matrix.matrix.zero_mat(weight_roll, 1) # bp
        self.__roll = weight_roll

        # 注意深拷贝，注意matrix类中下标从1开始计算
        # 将weight_mat中的数据拷贝入__weights中
        for i in range(1, weight_roll + 1):
            for j in range(1, weight_column + 1):
                self.__weights.setAt(i, j, weight_mat.readAt(i, j))

        # 将bias_mat中的数据拷贝到__bias中
        for i in range(1, bias_roll + 1):
            self.__bias.setAt(i, 1, bias_mat.readAt(i, 1))
        
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

    # 挤压函数ReLU
    @staticmethod
    def squeeze_ReLU(val):
        if (val <= 0):
            return 0
        else:
            return val

    # 挤压函数的导数ReLU
    @staticmethod
    def d_squeeze_ReLU(val):
        if (val <= 0):
            return 0
        else:
            return 1

    # Sigmoid
    def squeeze_Sigmoid(val):
        return 1 / (1 + math.exp(-1 * val))

    # der Sigmoid
    def d_squeeze_Sigmoid(val):
        return -1 * math.exp(-1 * val) / (1 + math.exp(-1 * val)) ** 2

    # 获得这一层的长度
    def getLength(self):
        return self.__roll

    # 通过输入进行计算
    # prev_layer 为matrix类
    def calcData(self, prev_layer_mat):
        self.__data = matrix.matrix.product(self.__weights, prev_layer_mat)
        self.__data = matrix.matrix.plus(self.__data, self.__bias)

        # 配置__z_mat矩阵，以便后期进行方向传播
        self.__z_mat = layer.__copy(self.__data)

        # 通过挤压函数挤压每个值
        for i in range(1, self.__roll + 1):
            self.__data.setAt(i, 1, layer.squeeze_Sigmoid(self.__data.readAt(i, 1)))

    # 返回数值
    def atWeight(self, rol, col):
        return self.__weights.readAt(rol, col)
    def atDelta(self, rol):
        return self.__delta.readAt(rol, 1)
    def atBias(self, rol):
        return self.__bias.readAt(rol, 1)
    def atZ(self, rol):
        return self.__z_mat.readAt(rol, 1)
    def atData(self, rol):
        return self.__data.readAt(rol, 1)

    # 返回权重矩阵
    def getWeithg_mat(self):
        return self.__weights

    # 返回数据矩阵
    def getData_mat(self):
        return self.__data

    # bp
    # 这里的参数都是layer类型
    def backpropogation(self, delta_next, weight_next, layer_prev):
        sigma = 0.1 # 学习效率i

        # 反向传播的delta传播
        for i in range(1, self.__roll + 1):
            for j in range(1, delta_next.getLength() + 1):
                self.__delta.appendAt(i, 1, weight_next.atWeight(j, i) * delta_next.atData(j)) # 先相加
            self.__delta.setAt(i, 1, d_squeeze_Sigmoid(self.__z_mat.readAt(i, 1))) # 再相乘

        # bp 对偏置b的修改
        for i in range(1, self.__roll + 1):
            self.__bias.appendAt(i, 1, -1 * self.__delta.readAt(i, 1) * sigma)

        # bp 对权重进行更改
        for j in range(1, weight_next.getLength() + 1):
            for i in range(1, self.__roll + 1):
                self.__weights.appendAt(j, i, -1 * self.__delta.readAt(j, 1) * layer_prev.atData(i) * sigma)

        pass

'''
wei = matrix.matrix([[1,2],[3,4],[5,6]])
bia = matrix.matrix([[1],[4],[2]])
ipt = matrix.matrix([[-7], [0]])

lay = layer(wei, bia)
lay.calcData(ipt)
'''
