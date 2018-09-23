import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import config
import ipdb
#ipdb.set_trace()
IM_WIDTH, IM_HEIGHT = 224, 224 #fixed size for InceptionV3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

def get_nb_files(directory):
	if not os.path.exists(directory):
		return 0
	cnt = 0
	for r, dirs, files in os.walk(directory):
		for dr in dirs:
			cnt += len(glob.glob(os.path.join(r, dr + '/*')))
	return cnt

def add_new_last_layer(base_model, nb_classes):
	#ipdb.set_trace()
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(FC_SIZE, activation='relu')(x)
	predictions = Dense(nb_classes, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	return model
def train(args):
	nb_train_samples = get_nb_files(args.train_dir)
	nb_classes = args.nb_classes
	nb_val_samples = get_nb_files(args.val_dir)
	nb_epoch = args.nb_epoch 
	batch_size = args.batch_size
	train_datagen = ImageDataGenerator(
		preprocessing_function = preprocess_input,
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)
	test_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=30,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)
	train_generator = train_datagen.flow_from_directory(
		args.train_dir,
		target_size=(IM_WIDTH, IM_HEIGHT),
		batch_size=batch_size)
	validation_generator = test_datagen.flow_from_directory(
		args.val_dir,
		target_size=(IM_WIDTH, IM_HEIGHT),
		batch_size=batch_size)
	base_model = VGG16(weights=None, include_top=False)
	model = add_new_last_layer(base_model, nb_classes)
	model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
	history_tl = model.fit_generator(
		train_generator,
		steps_per_epoch=nb_train_samples,
		epochs=nb_epoch,
		validation_data=validation_generator,
		validation_steps=nb_val_samples,)
if __name__ == '__main__':
	args = config.Config()
	train(args)
