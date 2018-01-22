import numpy as np
import tensorflow as tf
import scipy.io as sio
from glob import glob
import os
import math
import cv2

#============== Different Readers ==============
def load_train_data(args):


	paths = os.path.join(args.data, "training/*.jpeg")
	data_count = len(glob(paths))
	
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths))

	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)

	orig_images = tf.image.decode_jpeg(image_file, channels=3)
	gray_images = tf.image.rgb_to_grayscale(orig_images)

	size = 20
	orig_images = tf.image.resize_images(orig_images ,[args.input_height+size, args.input_width+size])
	gray_images = tf.image.resize_images(gray_images ,[args.input_height+size, args.input_width+size])

	orig_images = tf.random_crop(orig_images, [args.input_height, args.input_width, 3])
	gray_images = tf.random_crop(gray_images, [args.input_height, args.input_width, 1])

	orig_images = tf.image.random_flip_left_right(orig_images)
	gray_images = tf.image.random_flip_left_right(gray_images)

	orig_images = tf.image.convert_image_dtype(orig_images, dtype=tf.float32) / 255.#/ 127.5 - 1
	gray_images = tf.image.convert_image_dtype(gray_images, dtype=tf.float32) / 255.#/ 127.5 - 1


	orig, gray = tf.train.shuffle_batch([orig_images, gray_images],
						  				 batch_size=args.batch_size,
						  				 capacity=data_count*2,
						  				 min_after_dequeue=args.batch_size
						  				)

	return orig, gray, data_count



#load different datasets
def load_test_data(args):
	
	image = cv2.imread(args.test_img)
	image_3 = image
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	image = cv2.resize(image, (args.input_width, args.input_height))
	image = image[np.newaxis,...,np.newaxis]
	image = np.tile(image, [args.batch_size, 1, 1, 1]) / 255.

	image_3 = cv2.resize(image_3, (args.input_width, args.input_height))
	image_3 = image_3[np.newaxis,...]
	image_3 = np.tile(image_3, [args.batch_size, 1, 1, 1]) / 255.

	return image, image_3

def load_valid_data(args):
	paths = glob(os.path.join(args.data, "validation/*.jpg"))

	all_images = []
	all_images_L = []
	for p in paths:
		img = cv2.imread(p)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (args.input_width, args.input_height))
		all_images.append(img)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		all_images_L.append(img)

	all_images = np.array(all_images)
	all_images = all_images[...,np.newaxis] / 255.



	return all_images, np.tile(all_images, [1, 1, 1, 3])