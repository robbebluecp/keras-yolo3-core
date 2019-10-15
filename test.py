import yolo
import keras

model = yolo.DarkNet()(80, 9)
model.load_weights('/Users/yvan/stayby/keras-yolo3-core/model_data/yolov3.weights')


