"""
文本处理模块
"""

import numpy as np


def get_anchors(anchor_path='model_data/anchors.txt'):
    """

    :param anchor_path:     存放anchor的文件路径
    :return:                [[w1, h1], [w2, h2], ...]
    """
    with open(anchor_path) as f:
        text = f.read()
        f.close()
    anchors = text.split()
    anchors = list(map(lambda x: x.split(','), anchors))
    anchors = np.asarray(anchors, dtype=np.float)
    return anchors


def get_classes(class_path='../model_data/classes.txt'):
    """

    :param class_path:      存放类别的文件路径
    :return:                ['class1', 'class2', ...]
    """
    class_names = []
    with open(class_path, 'r') as f:
        items = f.readlines()
        for item in items:
            class_names.append(item.strip())
        f.close()
    return class_names


if __name__ == '__main__':
    anchors = get_anchors()
    print(anchors)
    class_names = get_classes()
    print(class_names)
