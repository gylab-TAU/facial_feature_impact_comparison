# check version of keras_vggface
import keras_vggface
import numpy as np
from tensorflow.keras.preprocessing import image
# from keras.utils.image_utils import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

# print version
print(keras_vggface.__version__)

model = VGGFace(model='resnet50')
# load the image
img = image.load_img(
    '/home/administrator/datasets/vggface2_mtcnn/n000388/0008_01.jpg',
    target_size=(224, 224))

# prepare the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = utils.preprocess_input(x, version=1)

# perform prediction
preds = model.predict(x)
print('Predicted:', utils.decode_predictions(preds))