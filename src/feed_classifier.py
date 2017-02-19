from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import os

batch_size = 32
nb_classes = 2
nb_epoch = 30

img_rows, img_cols = 32, 32
img_channels = 3


def load_data_generators():

	datagen = ImageDataGenerator(rescale=1./255)

	train_generator = datagen.flow_from_directory(
		'../data/train',
		target_size=(32, 32),
		batch_size=batch_size,
		class_mode='categorical')

	validation_generator = datagen.flow_from_directory(
		'../data/validation',
		target_size=(32, 32),
		batch_size=batch_size,
		class_mode='categorical')

	return train_generator, validation_generator

def get_model():
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, border_mode='same',
		                    input_shape=(32, 32, 3)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	return model

def main():
	if os.path.isfile('../models/first_model.h5'):
		model = load_model('../models/first_model.h5')
		model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

		for image in os.listdir('../data/demo'):
			if image[-4:] == '.jpg':
				im_matrix = imread(os.path.join('../data/demo',image))
				plt.imshow(im_matrix)
				plt.show()
				resized_im_matrix = imresize(im_matrix, (32, 32, 3))
				plt.imshow(resized_im_matrix)
				plt.show()
				print(model.predict(np.asarray([resized_im_matrix])))
	else:
		train_generator, validation_generator = load_data_generators()
		model = get_model()
		model.fit_generator(train_generator,
							# samples_per_epoch=1800,
							nb_epoch=nb_epoch,
							verbose=1,
							validation_data=validation_generator,
							nb_val_samples=200)
		
		model.save('../models/first_model.h5')

if __name__ == '__main__':
	main()