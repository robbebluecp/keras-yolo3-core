import numpy as np
import config
import keras.backend as K


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def iou_area_index(boxes, anchors):
    """

    :param boxes:       (N, 2) --- N x (w, h)
    :param anchors:     (N, 2) --- N x (w, h)
    :return:
    """
    boxes = np.expand_dims(boxes, -2)
    box_maxes = boxes / 2.
    box_mins = -box_maxes

    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    # anchor和box的交集
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # (x, 9)
    box_area = boxes[..., 0] * boxes[..., 1]
    # (x, 9)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    # 6!
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    # Find best anchor for each true box

    # (5, )
    best_anchor_index = np.argmax(iou, axis=-1)
    return best_anchor_index

def iou_cors_index(boxes, anchors):
    """

    :param boxes:       (N, 2) --- N x (x, y, w, h)
    :param anchors:     (N, 2) --- N x (x, y, w, h)
    :return:
    """
    # Expand dim to apply broadcasting.
    # (13, 13, 3, 1, 4)
    boxes = K.expand_dims(boxes, -2)
    # (13, 13, 3, 1, 2)
    boxes_xy = boxes[..., :2]
    # (13, 13, 3, 1, 2)
    boxes_wh = boxes[..., 2:4]
    boxes_wh_half = boxes_wh / 2.
    # (13, 13, 3, 1, 2)
    boxes_mins = boxes_xy - boxes_wh_half
    # (13, 13, 3, 1, 2)
    boxes_maxes = boxes_xy + boxes_wh_half

    # Expand dim to apply broadcasting.
    # (1, x, 4)
    anchors = K.expand_dims(anchors, 0)
    # (1, x, 2)
    anchors_xy = anchors[..., :2]
    # (1, x, 2)
    anchors_wh = anchors[..., 2:4]
    anchors_wh_half = anchors_wh / 2.
    # (1, x, 2)
    anchors_mins = anchors_xy - anchors_wh_half
    # (1, x, 2)
    anchors_maxes = anchors_xy + anchors_wh_half

    # (13, 13, 3, x, 2)
    intersect_mins = K.maximum(boxes_mins, anchors_mins)
    intersect_maxes = K.minimum(boxes_maxes, anchors_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    # (13, 13, 3, x)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = boxes_wh[..., 0] * boxes_wh[..., 1]
    b2_area = anchors_wh[..., 0] * anchors_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    # (13, 13, 3, x)
    return iou



def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """

    :param true_boxes:      (N, max_box, 5)
    :param input_shape:     (416, 416)
    :param anchors:         (9, 2)
    :param num_classes:     ...
    :return:
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # 中心点, (N, 20, 2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    # 宽高,   (N, 20, 2)
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 剔除0项
    valid_mask = boxes_wh[..., 0] > 0
    # 放缩
    # 从这里开始，对角坐标值被个替换成中心坐标值 + wh, 归一
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = true_boxes.shape[0]
    # [13, 26, 52]
    grid_shapes = [input_shape // config.scale_size[l] for l in range(num_layers)]
    # [(N, 13, 13, 3, 15), (N, 26, 26, 3, 15), (N, 52, 52, 3, 15)]
    y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(config.anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # 开始挑选最优anchor
    for N in range(batch_size):
        # Discard zero rows， 因为默认20个zeors_like, 一般都不够20的，有很多0填充项
        # (x, 2)
        wh = boxes_wh[N, valid_mask[N]]
        if len(boxes_wh) == 0:
            continue

        # 对于每个box，从9个anchor中，选出最佳anchor
        # (x, )
        best_anchor_indexes = iou_area_index(wh, anchors)

        for t, n in enumerate(best_anchor_indexes):
            for l in range(num_layers):
                if n in config.anchor_mask[l]:
                    # 中点y
                    i = np.floor(true_boxes[N, t, 0] * grid_shapes[l][1]).astype('int32')
                    # 中点x
                    j = np.floor(true_boxes[N, t, 1] * grid_shapes[l][0]).astype('int32')
                    #
                    k = config.anchor_mask[l].index(n)
                    c = true_boxes[N, t, 4].astype('int32')
                    y_true[l][N, j, i, k, 0:4] = true_boxes[N, t, 0:4]
                    y_true[l][N, j, i, k, 4] = 1
                    y_true[l][N, j, i, k, 5 + c] = 1
    # [(N, 13, 13, 3, 15), (N, 26, 26, 3, 15), (N, 52, 52, 3, 15)]
    return y_true


if __name__ == '__main__':
    np.random.seed(1)
    boxes = np.random.random_integers(0, 10, (20, 2))
    anchors = np.random.random_integers(0, 10, (9, 2))
    c = iou_area_index(boxes, anchors)
    print(c)
    boxes = np.random.random_integers(0, 10, (10, 10, 3, 2))
    anchors = np.random.random_integers(0, 10, (4, 2))
    c = iou_area_index(boxes, anchors)
    print(c.shape)