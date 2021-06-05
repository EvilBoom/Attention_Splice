# -*- coding:utf-8 -*-
# @File  : readt.py
# @Author: 张宝宇
# @Date  : 2021/6/5
# @Desc  :
import tqdm
if __name__ == '__main__':
    f = open('./A_thaliana_acc_all_examples.fasta')
    ls = []
    for line in f:
        if not line.startswith('>'):
            ls.append(line.replace('\n', ''))  # 去掉行尾的换行符真的很重要！
    print(1)
    f.close()
