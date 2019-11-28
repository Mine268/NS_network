#!/usr/bin/env python
# coding=utf-8

'''
这是一个矩阵的类，其中的矩阵从下标从1开始读取
拥有以下方法：
    矩阵声明           ([[...], [...], ...])
    矩阵加法           @plus(matrix, matrix)
    矩阵数乘           @multiply(number, matrix)
    矩阵乘法           @product(matrix_left, matrix_right)
    读取矩阵下标值     readAt(roll, colunm)
    设置矩阵下标的值   setAt(coll, colunm, value)
    获得矩阵大小       getSize()
    获取矩阵的转置     @Transposition()
    获取矩阵的字符版   toString()
    生成一个mn的零矩阵 @zero_mat(rol, column)
    // 带@的为静态方法
'''
class matrix(object):
    __size = [-1, -1] # 从一开始
    __mat = -1

    # 根据数组初始化
    def __init__(self, array):
        self.__mat = [[]]
        rol, col = len(array), len(array[0])
        for i in range(0, rol):
            for j in range(0, col):
                if (str(array[i][j]).isdigit() == True):
                    self.__mat[i].append(array[i][j])
                else:
                    raise Exception(print(f"输入中存在非数字。（array[{i}][{j}] == {array[i][j]}）"))
            self.__mat.append([])
        self.__mat.pop()

        self.__size = [rol, col]

    # 生成一个矩阵的字符形式
    def toString(self):
        res = ""
        for i in range(0, self.__size[0]):
            for j in range(0, self.__size[1]):
                res += " " + str(self.__mat[i][j])
            if (i != self.__size[0] - 1):
                res += "\n"
        return res

    # 返回矩阵的大小[rol, col]
    def getSize(self):
        return self.__size

    # 读取指定位置的数据
    def readAt(self, rol, col):
        matrix.__legalPos(self, rol, col)
        return self.__mat[rol - 1][col - 1]
    
    # 设置指定位置的数据
    def setAt(self, rol, col, val):
        matrix.__legalPos(self, rol, col)
        self.__mat[rol - 1][col - 1] = val
    
    # 矩阵的转置
    @staticmethod
    def Transposition(__mat):
        if (type(__mat) != matrix):
            raise Exception(print("只有matrix的类才能被转置。"))

        res = matrix.zero_mat(__mat.__size[1], __mat.__size[0])
        for i in range(0, __mat.__size[1]):
            for j in range(0, __mat.__size[0]):
                res.__mat[i][j] = __mat.__mat[j][i]
        return res

    # 检测下表是否合法
    @staticmethod
    def __legalPos(cls, rol, col):
        if (rol > cls.__size[0] or col > cls.__size[1]):
            raise Exception(print(f"无效的矩阵下标。（__size:{cls.__size[0]}x{cls.__size[1]} pos:{rol},{col}）"))
        else:
            pass

    # 生成一个rol行col列的零矩阵
    @staticmethod
    def zero_mat(rol, col):
        if (rol <= 0 or col <= 0):
            raise Exception(print("无法创建一个{rol}x{col}的矩阵。"))

        res = matrix([[]])
        for i in range(0, rol):
            for j in range(0, col):
                res.__mat[i].append(0)
            res.__mat.append([])
        res.__mat.pop()
        res.__size = [rol, col]
        return res

    # 矩阵相加
    @staticmethod
    def plus(__matA, __matB):
        if (__matA.__size[0] != __matB.__size[0] or __matA.__size[1] != __matB.__size[1]):
            raise Exception(print(f"非同型矩阵不能相加。（{__matA.__size[0]}x{__matA.__size[1]} {__matB.__size[0]}x{__matB.__size[1]}）"))
        else:
            res = matrix(__matA.__mat)
            for i in range(0, __matA.__size[0]):
                for j in range(0, __matA.__size[1]):
                    res.__mat[i][j] += __matB.__mat[i][j]
            return res

    # 矩阵数乘
    @staticmethod
    def multiply(num, __mat):
        res = matrix(__mat.__mat)
        for i in range(0, __mat.__size[0]):
            for j in range(0, __mat.__size[1]):
                res.__mat[i][j] *= num
        return res

    # 矩阵乘法
    @staticmethod
    def product(__matA, __matB):
        if (__matA.__size[1] != __matB.__size[0]):
            raise Exception(print(f"这两个矩阵不能相乘。（{__matA.__size[0]}x{__matA.__size[1]} {__matB.__size[0]}x{__matB.__size[1]}）"))
        else:
            res = matrix.zero_mat(__matA.__size[0], __matB.__size[1])
            for i in range(0, __matA.__size[0]):
                for j in range(0, __matB.__size[1]):
                    for k in range(0, __matA.__size[1]):
                        res.__mat[i][j] += __matA.__mat[i][k] * __matB.__mat[k][j]
            return res

