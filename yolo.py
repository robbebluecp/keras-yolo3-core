"""
yolo核心模块
1、DarkNet:      darknet网络模型
2、yolo_loss:    yolo损失函数
3、yolo_core:    yolo模型一个用用模块，用于对网络输出做放缩处理

"""

from keras.layers import *
import keras
from keras.regularizers import *
import numpy as np
import tensorflow as tf
import keras.backend as K
import config
from tools import utils


class DarkNet:
    def conv_base_block(self, inputs, filters, kernel_size, strides=(1, 1)):
        """
        darknet 自定义 conv层
        """
        if strides == (2, 2):
            padding = 'valid'
        else:
            padding = 'same'
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, kernel_regularizer=l2(5e-4))(inputs)
        return x

    def conv_block(self, inputs, filters, kernel_size, strides=(1, 1)):
        """
        darknet 基础组合层（conv + bn + leakyrelu）
        """
        x = self.conv_base_block(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    def res_block(self, inputs, filters, block_num):
        """
        darknet 基础残差块
        """
        x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        x = self.conv_block(inputs=x, filters=filters, kernel_size=3, strides=(2, 2))
        for i in range(block_num):
            y = self.conv_block(inputs=x, filters=filters // 2, kernel_size=1)
            y = self.conv_block(inputs=y, filters=filters, kernel_size=3)
            x = Add()([x, y])
        return x

    def output_block(self, inputs, filters, output_filters):
        """
        darknet输出层
        """
        x = self.conv_block(inputs=inputs, filters=filters, kernel_size=1)
        x = self.conv_block(inputs=x, filters=filters * 2, kernel_size=3)
        x = self.conv_block(inputs=x, filters=filters, kernel_size=1)
        x = self.conv_block(inputs=x, filters=filters * 2, kernel_size=2)
        x = self.conv_block(inputs=x, filters=filters, kernel_size=1)

        y = self.conv_block(inputs=x, filters=filters * 2, kernel_size=3)
        y = self.conv_base_block(inputs=y, filters=output_filters, kernel_size=1)
        return x, y

    def get_darknet(self,
                    n_class: int,
                    n_anchor: int):
        n_anchor = n_anchor // 3
        img = Input((None, None, 3))
        x = self.conv_block(inputs=img, filters=32, kernel_size=3)
        x = self.res_block(inputs=x, filters=64, block_num=1)
        x = self.res_block(inputs=x, filters=128, block_num=2)
        x = self.res_block(inputs=x, filters=256, block_num=8)
        x = self.res_block(inputs=x, filters=512, block_num=8)
        x = self.res_block(inputs=x, filters=1024, block_num=4)
        base_model = keras.models.Model(img, x)

        # o1
        x, y1 = self.output_block(inputs=x, filters=512, output_filters=n_anchor * (n_class + 5))

        x = self.conv_block(inputs=x, filters=256, kernel_size=1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, base_model.layers[152].output])
        # o2
        x, y2 = self.output_block(inputs=x, filters=256, output_filters=n_anchor * (n_class + 5))

        x = self.conv_block(inputs=x, filters=128, kernel_size=1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, base_model.layers[92].output])

        # o3
        x, y3 = self.output_block(inputs=x, filters=128, output_filters=n_anchor * (n_class + 5))

        model = keras.models.Model(img, [y1, y2, y3])
        return model

    def __call__(self,
                 n_class: int,
                 n_anchor: int,
                 *args, **kwargs):
        return self.get_darknet(n_class, n_anchor)



def yolo_loss(args, anchors, num_classes):
    '''Return yolo_loss tensor

    Parameters
    ----------
    y_pred_base: list of tensor, the output of yolo_body or tiny_yolo_body
                   [(N, 13, 13, n_anchor * (5 + n_class)),
                    (N, 26, 26, n_anchor * (5 + n_class)),
                    (N, 52, 52, n_anchor * (5 + n_class))],


    y_true: list of array, the output of preprocess_true_boxes
                   [(N, 13, 13, 3, 5+n_class),
                    (N, 26, 26, 3, 5+n_class),
                    (N, 52, 52, 3, 5+n_class)]
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    y_pred_base, y_true = args[:3], args[3:]
    # 3
    num_layers = len(anchors) // 3  # default setting
    # (9, 2)
    anchors = np.asarray(anchors, dtype=int)
    # (416, 416)
    input_shape = K.cast(config.image_input_shape, K.floatx())
    # [(13, 13), (26, 26), (52, 52)]
    grid_shapes = [K.cast(K.shape(y_pred_base[l])[1:3], K.floatx()) for l in range(num_layers)]
    loss = 0
    # N
    batch = K.shape(y_pred_base[0])[0]  # batch size, tensor

    batch_tensor = K.cast(batch, K.floatx())
    for l in range(num_layers):
        # 置信度
        # (N, 13, 13, 3, 1)
        object_mask = y_true[l][..., 4:5]
        # object_mask = K.cast(object_mask, 'bool')

        # 类别
        true_class_probs = y_true[l][..., 5:]

        # return (13, 13, 1, 2), (N, 13, 13, 3, 80), (N, 13, 13, 3, 2), (N, 13, 13, 3, 2)
        grid, raw_pred, pred_xy, pred_wh = yolo_core(y_pred_base[l],
                                                     anchors[config.anchor_mask[l]],
                                                     num_classes,
                                                     calc_loss=True)
        # (N, 13, 13, 3, 4)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # y_true是相对于416的归一，需要跟13、26、52进行比例放缩
        relative_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        relative_true_wh = K.log(y_true[l][..., 2:4] / anchors[config.anchor_mask[l]] * input_shape[::-1])
        relative_true_wh = K.switch(object_mask, relative_true_wh, K.zeros_like(relative_true_wh))  # avoid log(0)=-inf
        # (1 - 2)
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)

        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            # (13, 13, 3, 4) : (13, 13, 3)
            # return (x,  4)
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # (13, 13, 3, 4), (x, 4)
            # return (13, 13, 3, x)
            iou = utils.iou_cors_index(pred_box[b], true_box)
            # (13, 13, 3)
            best_iou = K.max(iou, axis=-1)
            # (13, 13, 3)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < config.ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        # if condition then body
        # if b < m, 执行loop_body, 初始参数为[0, ignore_mask]
        # (N, 13, 13, 3)
        _, ignore_mask = tf.while_loop(lambda b, *args: b < batch, loop_body, [0, ignore_mask])
        # (N, 13, 13, 3)
        ignore_mask = ignore_mask.stack()
        # (N, 13, 13, 3, 1)

        # ignore_mask = []
        # num = K.constant([0], dtype=tf.int32)
        # i = 0
        # print(num)
        # print(batch)
        # while K.less(num, batch):
        #     true_box = tf.boolean_mask(y_true[l][i, ..., 0:4], object_mask[i, ..., 0])
        #     iou = utils.iou_cors_index(pred_box[i], true_box)
        #     best_iou = K.max(iou, axis=-1)
        #     ignore_mask.append(K.cast(best_iou < config.ignore_thresh, dtype=K.floatx()))
        #     i += 1
        #     num += 1

        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(relative_true_xy, raw_pred[..., 0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(relative_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / batch_tensor
        wh_loss = K.sum(wh_loss) / batch_tensor
        confidence_loss = K.sum(confidence_loss) / batch_tensor
        class_loss = K.sum(class_loss) / batch_tensor
        loss += xy_loss + wh_loss + confidence_loss + class_loss
    return loss

def yolo_core(feats, anchors, num_classes, calc_loss=False):
    """

    :param feats:           (N, 13, 13, 3 * (5+n_class)), ...
    :param anchors:         (3, 2)
    :param num_classes:     15
    :param input_shape:     (416, 416)
    :param calc_loss:
    :return:
    """
    input_shape = config.image_input_shape
    # 3
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    # (1, 1, 1, 3, 2)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    # (13, 13)
    grid_shape = K.shape(feats)[1:3]  # height, width
    #
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    # (13, 13, 1, 2)
    grid = K.cast(grid, K.floatx())

    # (N, 13, 13, 3 * 15)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size.
    # 核心计算方法,
    # https://pjreddie.com/media/files/papers/YOLOv3.pdf   2.1
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        # (13, 13, 1, 2), (N, 13, 13, 3, 15), (N, 13, 13, 3, 2), (N, 13, 13, 3, 2)
        return grid, feats, box_xy, box_wh
    # (N, 13, 13, 3, 2), (N, 13, 13, 3, 2), (N, 13, 13, 3, 1), (N, 13, 13, 3, 10)
    return box_xy, box_wh, box_confidence, box_class_probs


if __name__ == '__main__':
    model = DarkNet()(10, 3)
    from keras.utils.vis_utils import plot_model

    plot_model(model, show_shapes=True, to_file='model_darknet.png')
