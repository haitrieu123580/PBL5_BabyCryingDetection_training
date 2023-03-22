import tensorflow as tf
from keras.models import model_from_json
with open('ModelThuNghiem\model8\cnn.json', 'r') as f:
    mymodel=model_from_json(f.read())

mymodel.load_weights("ModelThuNghiem\model8\cnn.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(mymodel)
tflite_model = converter.convert()
open('ModelThuNghiem\model8\model8.tflite','wb').write(tflite_model)