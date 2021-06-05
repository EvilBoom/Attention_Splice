# _*_ coding: utf-8 _*_
# @Time : 2021/4/15 22:07
# @Author : 张宝宇
# @Version：V 0.0
# @File : test.py
# @desc :
import tifffile

if __name__ == '__main__':
    filename = 'Corn Borner-M-3d-AL-11-.lsm'
    t = tifffile.TiffFile(filename)
    print(1)