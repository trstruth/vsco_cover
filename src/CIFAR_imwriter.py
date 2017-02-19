from keras.datasets import cifar10
from scipy.misc import imsave
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_negatives = X_train[:2500]

for idx, image in enumerate(X_negatives):
	imsave('../data/cifar10_files/{}.jpg'.format(idx), image)