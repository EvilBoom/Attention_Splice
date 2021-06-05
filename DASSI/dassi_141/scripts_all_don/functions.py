import numpy as np
import itertools
import pdb


def read_file(lines):
    ID = []
    Seq = []
    Raw_lab = []
    for i in range(len(lines)):
        if i % 3 == 0:
            ID.append(lines[i])
        if i % 3 == 1:
            Seq.append(lines[i])
        if i % 3 == 2:
            Raw_lab.append(lines[i])
    return ID, Seq, Raw_lab


# input：原始标签和对应的标签字典
# operation：将每个标签按照字典进行转化
# output：返回转换后的标签
def seq2num(seq, dic1):
    seq1 = []
    for s in seq:
        s1 = dic1[s]
        seq1.append(s1)
    return np.array(seq1)


def hilbert_curve(n):
    if n == 1:
        return np.zeros((1, 1), np.int32)
    t = hilbert_curve(n // 2)
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size * 2
    d = np.flipud(np.rot90(t, -1)) + t.size * 3
    return np.vstack(map(np.hstack, [[a, b], [d, c]]))


# input：样本，h_curve是一个142*1的列向量，1，a c g t 的 one-hot 字典
# operation：去重
# output：去重后的list
def plot_hb_dna(seq, h_curve, sub_length, map_dic):
    r, c = h_curve.shape  # r 142 c 1
    num_a = one_hot(seq, sub_length, map_dic)  # num-a 为样本的 one-hot 向量
    h_dna = np.zeros((r, c, 4 ** sub_length))  # 142,1,4 的零矩阵
    for i in range(len(num_a)):
        x, y = np.where(h_curve == i)  #
        h_dna[x, y, :] = num_a[i, :]
    return h_dna


def plot_row(seq,sub_length,map_dic):
    num_A = one_hot(seq, sub_length, map_dic)
    H_dna = -1.*np.ones((500, 4 ** sub_length))
    for i in range(len(num_A)):
        H_dna[i, :] = num_A[i, :]
    return H_dna

def plot_row1(seq,sub_length,map_dic):
    num_A = one_hot(seq, sub_length, map_dic)
    H_dna = -1.*np.ones((500, 1,4 ** sub_length))
    for i in range(len(num_A)):
        H_dna[i, 0, :] = num_A[i, :]
    return H_dna


# input：样本，1，a c g t 的 one-hot 字典
# operation：对样本进行向量空间的映射
# output：样本的向量映射再转 array
def one_hot(sequence, sub_len, mapping_dic):
    n_ = len(sequence)  # a c g t 数量
    sub_list = []  # 样本中的每一个存到sub——list中
    for i in range(n_ - sub_len + 1):
        sub_list.append(sequence[i:i + sub_len])
    res_ = []  # 将样本转换成向量形式
    for sub in sub_list:
        res_.append(mapping_dic[sub])
    return np.array(res_)  # 转换成 array


def cut(seq,sub_length):
    n = len(seq)
    new = []
    for i in range(n-sub_length+1):
        new.append(seq[i:i+sub_length])
    return np.array(new)


# input：list
# operation：去重
# output：去重后的list
def element(seq_list):
    list_ = []
    for s in seq_list:
        if s not in list_:
            list_.append(s)
    return list_


# input：list
# operation：去重
# output：去重后的list
def combination(elements, seq_length):
    keys = ['A', 'T', 'C', 'G']
    n_word = len(keys)
    array_word = np.eye(n_word)
    mapping_dic = {}
    for i in range(n_word):
        mapping_dic[keys[i]] = array_word[i, :]
    return mapping_dic

def diag_snake(m,n):
    H = np.zeros((m,n))
    count = 0
    for i in range(0,m+n-1):
        if i % 2 ==0:
            for x in range(m):
                for y in range(n):
                    if (x+y) == i:
                        H[x,y] = count
                        count +=1
        elif i%2 == 1:
            for x in range(m-1,-1,-1):
                for y in range(n):
                    if (x+y) == i:
                        H[x, y] = count
                        count += 1
    return H
def reshape_curve(m,n):
    return np.array((m*n)).reshape((m,n))
def snake_curve(m,n):
    H = np.arange(m*n).reshape((m,n))
    for i in range(m):
        if i%2==0:
            temp = H[i,:]
            H[i,:] = temp[::-1]
    return H
