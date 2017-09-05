import os
from os.path import join
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from classes import classes

base_dir = os.getcwd()
trained_dir = join(base_dir, 'trained')
classes_path = join(trained_dir, 'classes-resnet50')
model_path = join(trained_dir, 'model-resnet50.h5')
fine_tuned_weights_path = join(trained_dir, 'fine-tuned-resnet50-weights.h5')

img_path = '/data/flower/keras-transfer-learning-for-oxford102/data/jpg/image_04095.jpg'

def imread(path, size=(224,224)):
    x = image.load_img(path, target_size=size)
    return image.img_to_array(x)

def load_imgs(paths):
    x = [imread(path) for path in paths]
    return preprocess_input(np.array(x))

model = load_model(model_path)

from sklearn.externals import joblib
remap = joblib.load(classes_path)

def predict(paths):
    y = model.predict(load_imgs(paths))
    ans_ix = y.argsort(1)[:, -5:]
    ans = [[{
        'class': classes[int(remap[i])],
        'score': y[j][i]
    } for i in ans_ix[j]] for j in range(len(ans_ix))]

    return ans

