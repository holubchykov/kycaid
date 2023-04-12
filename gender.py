import tensorflow as tf
from deepface.basemodels import VGGFace


tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

labels = ["Woman", "Man"]


def loadModel():

    model = VGGFace.baseModel()
    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation("softmax")(base_model_output)
    gender_model = Model(inputs=model.input, outputs=base_model_output)
    gender_model.load_weights("weights/gender_model_weights.h5")

    return gender_model