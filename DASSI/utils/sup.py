# -*- coding:utf-8 -*-
# @File  : sup.py
# @Author: 张宝宇
# @Date  : 2021/6/5
# @Desc  :
# train 中的摘出来的辅助函数
import datetime

def batchify1(data, seq_len, bsz, args):
    nbatch = data.size(0) // (seq_len * bsz)
    data = data.narrow(0, 0, nbatch * bsz * seq_len)
    data = data.view(bsz * nbatch, seq_len).t().contiguous()
    print(data.size())
    data = data.cuda()
    return data


def common_data(list1, list2):
    result = 0
    for x in list1:
        for y in list2:
            if x == y:
                result = result + 1
    return result


if __name__ == '__main__':
    # common——data 找寻list中的相同元素个数，如果不重复可以使用set 来简化操作
    a = [1, 2, 3, 4, 5]
    b = [5, 6, 7, 8, 9]
    start = datetime.datetime.now()
    print(common_data(a, b))
    end = datetime.datetime.now()
    print((end - start).microseconds)
    start = datetime.datetime.now()
    print(len(set(a)&set(b)))
    end = datetime.datetime.now()
    print((end - start).microseconds)
    a = [1, 2, 3, 4, 5]
    b = [6, 7, 8, 9]
    print(common_data(a, b))
