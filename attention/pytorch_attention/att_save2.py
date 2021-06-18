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
from numpy.core.fromnumeric import *

s_dict = {'A': '0', 'C': '1', 'G': '2', 'T': '3'}
dict_S = {'0': 'A', '1': 'C', '2': 'G', '3': 'T'}
one_hot_dict = {
    0: [0, 0, 0, 1],
    1: [0, 0, 1, 0],
    2: [0, 1, 0, 0],
    3: [1, 0, 0, 0]
}
hot_one_dict = {
    '0001': 0,
    '0010': 1,
    '0100': 2,
    '1000': 3
}


# @nb.jit()
def test3(x):
    x = np.array(x, dtype='float64').reshape(1, -1)
    return Normalizer(norm='max').fit_transform(x)[0].tolist()


def cluster():
    with open('./attention_sample.json') as f:
        data = json.load(f)
    size = 9
    seq_set, result_list = set(), []  # 8位序列集合
    dd_dict = []
    for idx, item in enumerate(data):
        attention, inputs, att_size_list = item['attention'], item['inputs'], []
        for i in range(0, len(attention), 1):
            if i == 132:
                break
            temp_attention, temp_input = attention[i:i + size], inputs[i:i + size]
            sum_score = sum(temp_attention)
            if sum_score < 0.1:
                break
            temp_dict = {
                'index': i,
                'idx': idx,
                'input': temp_input,
                'score': sum_score,
                # 'attention':temp_attention
            }
            # attention   normalize 过了 [1*9]
            temp_attention = np.array(test3(temp_attention)).reshape(-1, 1)
            # temp_input 1*9 => 4*9  # 不用矩阵了，效果不行
            temp_input = np.array([one_hot_dict[temp_i] for temp_i in temp_input])
            cluster_input = np.concatenate((temp_input, temp_attention), axis=1)
            # temp_input = [one_hot_dict[temp_i] for temp_i in temp_input]
            position = np.array([i for i in range(i, i + size)]).reshape(-1, 1)
            cluster_input = np.concatenate((cluster_input, position), axis=1).T
            cluster_input_list = cluster_input.tolist()
            # temp_dict = {
            #     'index': i,  # 对应原始输入序列的位置
            #     'attention_sum': sum_score,  # attention 加和
            #     'attention': temp_attention,
            #     'inputs': temp_input,  # 按照滑动窗口抽取的输入序列—A T T G A C T...
            #     # 转换成 字典映射，就是一个数
            # }

            # seq_set.add(''.join(str(e) for e in temp_input))
            att_size_list.append(cluster_input_list)
            dd_dict.append(temp_dict)
        result_list.extend(att_size_list)
    input_clusters = np.array(result_list)
    # A = input_clusters[0]
    # B = input_clusters[1]
    # sim_sample = cos_sim(A, B)
    # sim_sample2 = mtx_similar1(A, B)
    # sim_list = []
    # for i in range(0,len(input_clusters)-1,1):
    #     for j in range(i+1, len(input_clusters), 1):
    #         # sim_list.append(mtx_similar1(input_clusters[i], input_clusters[j]))
    #         sim_list.append(mtx_similar2(input_clusters[i], input_clusters[j]))
    # a = np.array(sim_list).T
    # print(sim_sample, sim_sample2)

    C = DBSCAN(input_clusters, 3, 5)
    for key, values in C.items():
        for ids, v in enumerate(values):
            values[ids] = dd_dict[v]
        C[key] = values
    save_database(C)
    print(1)


# 获取一个点的ε-邻域（记录的是索引）
def getNeibor(data, dataSet, e):
    res = []
    for i in range(shape(dataSet)[0]):
        if mtx_similar2(data, dataSet[i]) < e:
            res.append(i)
    return res


# 计算两个向量之间的欧式距离
def calDist(X1, X2):
    sum = 0
    for x1, x2 in zip(X1, X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5


def cos_sim(A, B):
    similarity = np.dot(A, B.T)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    return cosine.T * inv_mag


def mtx_similar2(arr1: np.ndarray, arr2: np.ndarray) -> float:
    sub = arr1 - arr2
    x_norm = np.linalg.norm(sub, ord=2)
    return x_norm


def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
    注意有展平操作。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:实际是夹角的余弦值，ret = (cos+1)/2
    """
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1 ** 2) * np.sum(farr2 ** 2))
    similar = numer / denom  # 这实际是夹角的余弦值
    return (similar + 1) / 2  # 姑且把余弦函数当线性


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


def save_database(result_dict):
    # 存到数据库里
    connection = pymysql.connect(
        host="39.106.192.80",
        user="root",
        port=3307,
        password="root",
        database='dna'
    )
    with connection:
        with connection.cursor() as cursor:
            # Create a new record
            for key, values in result_dict.items():
                for j in values:
                    str_input = ''.join(dict_S[str(e)] for e in j['input'])
                    sql = f"INSERT INTO `data` (`cluster`, `seq_id`,`start`, `input`, `att_score`) " \
                          f"VALUES ({key}, {j['index']}, {j['idx']},'{str_input}', {j['score']:.5f})"
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
