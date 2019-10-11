import numpy as np
from tools import utils_image, utils


def data_generator(label_lines, batch_size, input_shape, anchors, num_classes):
    """

    :param annotation_lines:    /xxx/VOCdevkit/VOC2007/JPEGImages/000017.jpg 185,62,279,199,14 90,78,403,336,12
    :param batch_size:          假设：32
    :param input_shape:         (416, 416)
    :param anchors:             9 x 2
    :param num_classes:         假设：10
    :return:
    """

    n = len(label_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(label_lines)

            # 处理数据
            label_line = label_lines[i]
            info = label_line.split()
            image_file_path, cors = info[0], info[1:]
            cors = np.array([np.array(list(map(int, box.split(',')))) for box in cors])

            # 数据增强
            # (416, 416, 3), (20, 5)
            image, box = utils_image.augument(image_file_path, cors)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        # (N, 20, 5), (416, 416), (9, 2), 10
        # return [N, 13, 13, 3, 15]
        y_true = utils.preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)
