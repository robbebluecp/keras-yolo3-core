from tools import utils_text
import config
import yolo
from keras.layers import Lambda, Input
from keras.optimizers import Adam
from keras import Model
from generator import data_generator


anchors = utils_text.get_anchors(config.anchor_file_path)
class_names = utils_text.get_classes(config.class_file_path)
num_anchors = len(anchors)
num_classes = len(class_names)
class_mapping = dict(enumerate(class_names))
class_mapping = {class_mapping[key]: key for key in class_mapping}

f = open(config.label_file_path)
label_lines = f.readlines()

train_lines = label_lines[:int(len(label_lines) * config.validation_split)]
valid_lines = label_lines[int(len(label_lines) * config.validation_split):]

model_yolo = yolo.DarkNet()(n_class=num_classes, n_anchor=num_anchors)

h, w = config.image_input_shape
y_true = [Input(shape=(h // config.scale_size[l], w // config.scale_size[l], num_anchors // 3, num_classes + 5)) for l in range(3)]
model_loss = Lambda(yolo.yolo_loss, output_shape=(1,), name='yolo_loss',
                    arguments={'anchors': anchors, 'num_classes': num_classes})(
    [*model_yolo.output, *y_true])

model = Model([model_yolo.input, *y_true], model_loss)
model.compile(optimizer=Adam(1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
model.fit_generator(generator=data_generator(label_lines=train_lines,
                                             batch_size=config.batch_size,
                                             input_shape=config.image_input_shape,
                                             anchors=anchors,
                                             num_classes=num_classes),
                    validation_data=data_generator(label_lines=valid_lines,
                                             batch_size=config.batch_size,
                                             input_shape=config.image_input_shape,
                                             anchors=anchors,
                                             num_classes=num_classes),
                    steps_per_epoch=len(label_lines) // config.batch_size,
                    validation_steps=int(len(label_lines) * config.validation_split),
                    epochs=config.epochs
                    )
