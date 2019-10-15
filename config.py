"""
config
"""

# anchor 文件路径
anchor_file_path = '/Users/yvan/stayby/keras-yolo3-core/model_data/anchors.txt'
# class文件路径
class_file_path = '/Users/yvan/stayby/keras-yolo3-core/model_data/voc_classes.txt'
# 样本数据文件路径
label_file_path = '/Users/yvan/stayby/keras-yolo3-core/sample_data/train.txt'
# 样式路径
font_path = '/Users/yvan/stayby/keras-yolo3-core/font_data/FiraMono-Medium.otf'
# 默认输入图像大小
image_input_shape = (416, 416)
# anchor使用顺序
anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# 放缩比例
scale_size = [32, 16, 8]
# 数据增强参数
max_boxes = 20
jitter = 0.3
hue = 0.1
sat = 1.5
val = 1.5
# iou阈值
ignore_thresh = 0.5

batch_size = 8
validation_split = 0.1
epochs = 3