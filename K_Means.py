import random
import sys

import numpy as np
from scipy import spatial

import globals as g

X = g.vectors
K = 35
print(X.__len__())


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# comment this line to remove linear scaling
# X = NormalizeData(X)

cn_index = random.sample(range(0, len(X)), K)
print("random samples chosen : ", cn_index)

cn = []
for i in cn_index:
    cn.append(X[i])

changed = True
max_itr = 300

cluster = [-1 for _ in range(len(X))]
old_cluster = [-1 for _ in range(len(X))]
while changed and max_itr > 0:
    calc = [[] for _ in range(K)]
    for i in range(K):
        for j in range(len(X)):
            calc[i].append(spatial.distance.euclidean(cn[i], X[j]))
            # use spacial.distance.cityblock for manhattan distance

    for i in range(len(X)):
        minimum = sys.maxsize
        min_index = -1
        for j in range(K):
            if calc[j][i] < minimum:
                minimum = calc[j][i]
                min_index = j
        cluster[i] = min_index
    changed = False
    if old_cluster != cluster:
        changed = True
        old_cluster = cluster
    for i in range(len(cn)):
        n_of_c = 0
        total = [0 for _ in range(len(X[0]))]
        for j in range(len(cluster)):
            if cluster[j] == i:
                n_of_c = n_of_c + 1
                for k in range(len(X[0])):
                    total[k] = total[k] + X[j][k]
        old_c = cn[i].copy()
        for l in range(len(X[0])):
            cn[i][l] = total[l] / n_of_c
    print(cluster)
    max_itr = max_itr - 1
print()

# noinspection PyUnboundLocalVariable
print("final cluster classification = ", cluster)
target_cluster = cluster[g.index]
list_index = []
c = 0
for i in cluster:
    if i == target_cluster:
        list_index.append(c)
    c += 1
for x in list_index:
    print(g.original_text_train[x])
