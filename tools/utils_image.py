from .utils import rand
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import config


def augument(image_file_path, cors):
    """

    :param image_file_path:
    :param cors:
    :param max_boxes:
    :param jitter:
    :param hue:
    :param sat:
    :param val:
    :param proc_img:
    :return:
    """

    image = Image.open(image_file_path)
    # PIL顺序为(w, h)
    # 原始大小
    rw, rh = image.size
    h, w = config.image_input_shape

    # resize image
    scale1 = w / h * rand(1 - config.jitter, 1 + config.jitter) / rand(1 - config.jitter, 1 + config.jitter)
    # scale1 = np.sqrt(np.random.rand() + 0.5)
    scale2 = rand(.25, 2)
    # scale2 = np.square(np.random.rand() + 0.5)
    # 新的大小
    if scale1 < 1:
        nh = int(scale2 * h)
        nw = int(nh * scale1)
    else:
        nw = int(scale2 * w)
        nh = int(nw / scale1)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 颜色微调
    hue = rand(-config.hue, config.hue)
    sat = rand(1, config.sat) if rand() < .5 else 1 / rand(1, config.sat)
    val = rand(1, config.val) if rand() < .5 else 1 / rand(1, config.val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # 修正坐标 boxes
    cors_data = np.zeros((config.max_boxes, 5))
    if len(cors) > 0:
        np.random.shuffle(cors)
        cors[:, [0, 2]] = cors[:, [0, 2]] * nw / rw + dx
        cors[:, [1, 3]] = cors[:, [1, 3]] * nh / rh + dy
        if flip:
            cors[:, [0, 2]] = w - cors[:, [2, 0]]
        cors[:, 0:2][cors[:, 0:2] < 0] = 0
        cors[:, 2][cors[:, 2] > w] = w
        cors[:, 3][cors[:, 3] > h] = h
        box_w = cors[:, 2] - cors[:, 0]
        box_h = cors[:, 3] - cors[:, 1]
        box = cors[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > config.max_boxes:
            box = box[:config.max_boxes]
        cors_data[:len(box)] = box

    # 归一的image， 未归一的修正过的cors
    # (416, 416, 3), (20, 5)
    return image_data, cors_data

