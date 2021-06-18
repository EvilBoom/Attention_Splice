# _*_ coding: utf-8 _*_
# @Time : 2021/6/17 9:19
# @Author : 张宝宇
# @Version：V 0.0
# @File : att_save.py
# @desc :
# 存储attention
import json
import numpy.random as random
import numpy as np
import pymysql
# import numba as nb
from sklearn.datasets import load_digits
from sklearn.preprocessing import Normalizer, MinMaxScaler
from  numpy.core.fromnumeric import *
s_dict = {'A': '0', 'C': '1', 'G': '2', 'T': '3'}
dict_S = {'0': 'A', '1': 'C', '2': 'G', '3': 'T'}
one_hot_dict = {
    0: [0, 0, 0, 1],
    1: [0, 0, 1, 0],
    2: [0, 1, 0, 0],
    3: [1, 0, 0, 0]
}


def test():
    x = np.array([0.001, 0.2, 0.0003, 0.04], dtype='float64').reshape(1, -1)
    print("Before normalization: ", x)
    options = ['l1', 'l2', 'max']
    for opt in options:
        norm_x = Normalizer(norm=opt).fit_transform(x)
        print(type(norm_x), norm_x[0].tolist())
        print("After %s normalization: " % opt.capitalize(), norm_x)


def test2():
    scaler = MinMaxScaler(feature_range=(0.1, 1))
    x = np.array([0.001, 0.2, 0.0003, 0.04], dtype='float64').reshape(1, -1)
    print(x)
    print(scaler.fit_transform(x))


# @nb.jit()
def test3(x):
    x = np.array(x, dtype='float64').reshape(1, -1)
    return Normalizer(norm='max').fit_transform(x)[0].tolist()


def cluster():
    with open('./attention_sample.json') as f:
        data = json.load(f)
    size = 9
    seq_set, result_list = set(), []  # 8位序列集合
    for idx, item in enumerate(data):
        attention, inputs, att_size_list = item['attention'], item['inputs'], []
        for i in range(0, len(attention), 1):
            if i == 132:
                break
            temp_attention, temp_input = attention[i:i + size], inputs[i:i + size]
            sum_score = sum(temp_attention)
            if sum_score < 0.5:
                break
            # attention   normalize 过了 [1*9]
            temp_attention = test3(temp_attention)
            # temp_input 1*9 => 4*9
            temp_input = np.array([one_hot_dict[temp_i] for temp_i in temp_input])
            position = range(i, i + size)
            cluster_input = temp_attention.extend(temp_input).extend(position)
            # temp_dict = {
            #     'index': i,  # 对应原始输入序列的位置
            #     'attention_sum': sum_score,  # attention 加和
            #     'attention': temp_attention,
            #     'inputs': temp_input,  # 按照滑动窗口抽取的输入序列—A T T G A C T...
            #     # 转换成 字典映射，就是一个数
            # }
            temp_dict = {
                'index': i,  # 对应原始输入序列的位置
                'input': cluster_input
            }
            # seq_set.add(''.join(str(e) for e in temp_input))
            att_size_list.append(temp_dict)
        result_list.append(att_size_list)
    C = DBSCAN(result_list,)


# 获取一个点的ε-邻域（记录的是索引）
def getNeibor(data, dataSet, e):
    res = []
    for i in range(shape(dataSet)[0]):
        if calDist(data, dataSet[i]) < e:
            res.append(i)
    return res


# 计算两个向量之间的欧式距离
def calDist(X1, X2):
    sum = 0
    for x1, x2 in zip(X1, X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5


# 密度聚类算法
def DBSCAN(dataSet, e, minPts):
    coreObjs = {}  # 初始化核心对象集合
    C = {}
    n = shape(dataSet)[0]
    # 找出所有核心对象，key是核心对象的index，value是ε-邻域中对象的index
    for i in range(n):
        neibor = getNeibor(dataSet[i], dataSet, e)
        if len(neibor) >= minPts:
            coreObjs[i] = neibor
    oldCoreObjs = coreObjs.copy()
    k = 0  # 初始化聚类簇数
    notAccess = list(range(n))  # 初始化未访问样本集合（索引）
    while len(coreObjs) > 0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = coreObjs.keys()
        # 随机选取一个核心对象
        randNum = random.randint(0, len(cores))
        cores = list(cores)
        core = cores[randNum]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue) > 0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys():
                delte = [val for val in oldCoreObjs[q] if val in notAccess]  # Δ = N(q)∩Γ
                queue.extend(delte)  # 将Δ中的样本加入队列Q
                notAccess = [val for val in notAccess if val not in delte]  # Γ = Γ\Δ
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    return C


def specture():
    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size
    print(1)


def save_database(result_list):
    # 存到数据库里
    connection = pymysql.connect(
        host="localhost",
        user="root",
        password="zby6250196",
        database='dna'
    )
    with connection:
        with connection.cursor() as cursor:
            # Create a new record
            for ids, item in enumerate(result_list):
                for j in item:
                    str_input = ''.join(dict_S[str(e)] for e in j['inputs'])
                    sql = f"INSERT INTO `data` (`id`, `index`, `input`, `att_score`) VALUES ({ids}, {j['index']}, '{str_input}', {j['attention']:.5f})"
                    cursor.execute(sql)
        # connection is not autocommit by default. So you must commit to save
        # your changes.
        connection.commit()


if __name__ == '__main__':
    # test()
    # test2()
    # specture()
    cluster()
    print(1)
