"""
https://github.com/lars76/kmeans-anchor-boxes

某大佬对anchor进行聚类的计算逻辑
"""

import numpy as np


def iou(box, clusters):
    """

    仅与面积相关的IOU计算逻辑

    :param box:             1 x 2
    :param clusters:        9 x 2
    :return:                1 x 9
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def kmeans(boxes, k=9):
    """

    跟IOU相关的聚类算法

    :param boxes:   N x 2
    :param k:       9
    :return:        9 x 2
    """
    # N
    rows = boxes.shape[0]

    # N * 9
    distances = np.empty((rows, k))
    # N
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # 9 * 2
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        # N
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = np.median(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters
    if not np.all(clusters * 416):
        print('again', clusters)
        return kmeans(boxes, k)
    return np.asarray(clusters * 416, dtype=np.int)


if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.random_integers(1, 50, size=(20, 2))
    b = kmeans(a, 9)
    with open('model_data/anchor_kmeans.txt', 'w') as f:
        b = np.sort(b, 0)
        print(b)
        for i in b:
            i = list(map(str, i))
            f.write(','.join(i))
            f.write('\n')
        f.close()
