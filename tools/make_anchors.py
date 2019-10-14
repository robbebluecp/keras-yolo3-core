import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import config


def get_centers(inputs, is_show=False):
    K = KMeans(n_clusters=9)
    K.fit_predict(inputs)
    if is_show:
        plt.scatter(inputs[:, 0], inputs[:, 1], marker='o')
        plt.show()
        print(K.cluster_centers_)
        y = K.fit_predict(inputs)
        plt.scatter(inputs[:, 0], inputs[:, 1], c=y)
        plt.show()
    return K.cluster_centers_


result = []
f = open(config.label_file_path, 'r')
items = f.readlines()
for item in items:
    boxes = item.strip().split(' ')[1:]
    for box in boxes:
        cors = box.split(',')
        cors = list(map(int, cors[:-1]))
        result.append(cors)

result = list(map(lambda x: [x[3] - x[1], x[2] - x[0]], result))
result = np.asarray(result)

a = get_centers(result, True)
