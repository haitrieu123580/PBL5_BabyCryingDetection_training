import tensorflow as tf
from keras.models import model_from_json
with open('ModelThuNghiem\model3\cnn.json', 'r') as f:
    mymodel=model_from_json(f.read())

mymodel.load_weights("ModelThuNghiem\model3\cnn.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(mymodel)
tflite_model = converter.convert()
open('ModelThuNghiem\model3\model3.tflite','wb').write(tflite_model)