import numpy as np
import tensorflow as tf
import scipy.io as sio
from glob import glob
import os
import math
import cv2

#============== Different Readers ==============
def images_reader(args):


	paths = os.path.join(args.datapath, "flowers/*.png")
	data_count = len(glob(paths))
	
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths))

	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)

	orig_images = tf.image.decode_png(image_file, channels=3)
	gray_images = tf.image.rgb_to_grayscale(orig_images)
	# gray_images = tf.image.grayscale_to_rgb(gray_images)


	orig_images = tf.image.resize_images(orig_images ,[args.input_width, args.input_height]) / 255.
	gray_images = tf.image.resize_images(gray_images ,[args.input_width, args.input_height]) / 255.


	orig, gray = tf.train.shuffle_batch([orig_images, gray_images],
						  				 batch_size=args.batch_size,
						  				 capacity=data_count,
						  				 min_after_dequeue=args.batch_size
						  				)

	return orig, gray, data_count




#load different datasets
def load_data(args):
	
	orig, gray, data_count = images_reader(args)

	return orig, gray, data_count 