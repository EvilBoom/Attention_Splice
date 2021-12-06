# _*_ coding: utf-8 _*_
# @Time : 2021/6/17 9:19
# @Author : 张宝宇
# @Version：V 0.0
# @File : att_save.py
# @desc :
# 存储attention
import json

import numpy as np
import numpy.random as random
# import pymysql
# import numba as nb
import pymysql
from numpy.core.fromnumeric import *
from sklearn.preprocessing import Normalizer

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
    with open('./sas_attention_sample.json') as f:
        data = json.load(f)
    data = data[:300]
    size = 15
    seq_set, result_list = set(), []  # 8位序列集合
    dd_dict = []
    # position=[]
    # for p in range(140-size+1):
    #     position.append(p*2/(140-size+1))
    # print(position)
    for idx, item in enumerate(data):
        attention, inputs, att_size_list = item['attention'], item['inputs'], []
        attention = test3(attention)
        position = [i for i in range(140)]
        position = test3(position)
        for i in range(0, len(attention), 1):
            if i == 140 - size + 1:
                break
            temp_attention, temp_input, pos = attention[i:i + size], inputs[i:i + size], position[i:i + size]
            sum_score = sum(temp_attention)
            if sum_score < 8:
                break
            str_input = ''.join(dict_S[str(e)] for e in temp_input)
            temp_dict = {
                'index': i,
                'idx': idx,
                'input': str_input,
                'score': sum_score,
                # 'attention':temp_attention
            }
            # attention   normalize 过了 [1*9]
            temp_attention = np.array(temp_attention).reshape(-1, 1)
            # temp_input 1*9 => 4*9
            temp_input = np.array([one_hot_dict[temp_i] for temp_i in temp_input])
            cluster_input = np.concatenate((temp_input, temp_attention), axis=1)
            pos = np.array(pos).reshape(-1, 1)
            cluster_input = np.concatenate((cluster_input, pos), axis=1).T
            cluster_input_list = cluster_input.tolist()
            att_size_list.append(cluster_input_list)
            dd_dict.append(temp_dict)
        result_list.extend(att_size_list)
    input_clusters = np.array(result_list)
    C = DBSCAN(input_clusters, 3.5, 10)
    for key, values in C.items():
        for ids, v in enumerate(values):
            values[ids] = dd_dict[v]
        C[key] = values
    save_database(C)
    # input_clusters = input_clusters[:300]
    # centroids, clusterAssment = KMeans(input_clusters, 10)
    # m, n = clusterAssment.shape
    # for i in range(m):
    #     sequence = input_clusters[m - 1, 0:4, :]
    #     attention = input_clusters[m - 1, 5, :]
    #     position = input_clusters[m - 1, 6, :]
    #     print(input_clusters[m - 1, 6, :])

    print(1)
    print(1)


# 获取一个点的ε-邻域（记录的是索引）
def getNeibor(data, dataSet, e):
    res = []
    for i in range(shape(dataSet)[0]):
        if mtx_similar2(data, dataSet[i]) < e:
            res.append(i)
    return res


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
    input_1, input_2 = arr1[:4], arr2[:4]
    input_dis = np.linalg.norm(input_1 - input_2, ord=2)
    atten_1, atten_2 = arr1[4:5], arr2[4:5]
    att_dis = np.linalg.norm(atten_1 - atten_2, ord=2)
    pos_1, pos_2 = arr1[-1:], arr2[-1:]
    pos_dis = np.linalg.norm(pos_1 - pos_2, ord=2)
    x_norm = input_dis * 2 + att_dis * 1.5 + pos_dis
    return x_norm


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
            for key, values in result_dict.items():
                for j in values:
                    # str_input = ''.join(dict_S[str(e)] for e in j['input'])
                    sql = f"INSERT INTO `data` (`cluster`, `start`,`seq_id`, `input`, `att_score`) " \
                          f"VALUES ({key}, {j['index']}, {j['idx']},'{j['input']}', {j['score']:.5f})"
                    cursor.execute(sql)
        connection.commit()


# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    # 获取样本数与特征值
    m, n, z = dataSet.shape  # 把数据集的行数和列数赋值给m,n
    # 初始化质心,创建(k,n)个以零填充的矩阵
    centroids = np.zeros((k, n, z))
    # 循环遍历特征值
    print(n)
    print(z)
    for i in range(k):
        index = np.random.uniform(0, 1, (n, z))

        # 计算每一列的质心,并将值赋给centroids
        centroids[i, :, :] = index
        # 返回质心
    return centroids


def KMeans(dataSet, k):
    print(dataSet.shape)
    m, n, z = dataSet.shape
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True

    # 创建质心,随机K个质心
    centroids = randCent(dataSet, k)
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    ite = 0
    while clusterChange:
        clusterChange = False

        # 遍历所有样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有数据找到距离每个点最近的质心,
            # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distance = distEclud(centroids[j, :, :], dataSet[i, :, :])
                # distance=mtx_similar2(centroids[j,:,:],dataSet[i,:,:])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)
                clusterAssment[i, :] = minIndex, minDist
            ite = ite + 1
            # print(ite)

        # 遍历所有质心并更新它们的取值
        for j in range(k):
            # 通过数据过滤来获得给定簇的所有点
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[j, :, :] = np.mean(pointsInCluster, axis=0)

    print("Congratulation,cluster complete!")
    # 返回所有的类质心与点分配结果
    print(clusterAssment)

    return centroids, clusterAssment


if __name__ == '__main__':
    # test()
    # test2()
    # specture()
    cluster()
    print(1)
