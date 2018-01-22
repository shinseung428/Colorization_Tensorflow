import tensorflow as tf
import numpy as np
import os

from DataReader import *
from utils import *

class network():
	def __init__(self, args):

		#load image data X and label data Y

		if args.test_img == "":
			self.orig, self.gray, self.data_count = load_train_data(args)
		else:
			# self.orig, self.gray, self.data_count = load_test_data(args)
			self.orig = tf.placeholder(tf.float32, [args.batch_size, args.input_height, args.input_width, 3], name="orig")
			self.gray = tf.placeholder(tf.float32, [args.batch_size, args.input_height, args.input_width, 1], name="gray")
			self.data_count = args.batch_size

		self.build_model()
		self.build_loss()

		#summary
		self.loss_sum = tf.summary.scalar("loss", self.loss)
		#GT
		self.input_img_sum = tf.summary.image("input_img", self.orig)
		self.gray_img_sum = tf.summary.image("gray_img", self.gray)
		self.gt_lab_sum = tf.summary.image("gt_lab", self.gt_lab)
		#pred
		self.pred_lab_sum = tf.summary.image("pred_lab", self.pred_lab)
		self.pred_rgb_sum = tf.summary.image("pred_rgb", self.pred_rgb)

	def build_model(self):
		low_out, low_nets = self.low_level_features_network(self.gray, name="low_level_features_network")

		low_out_scaled, low_nets_scaled = self.low_level_features_network(self.gray, reuse=True, name="low_level_features_network")

		middle_out, middle__nets = self.middle_level_features_network(low_out, name="middle_level_features_network")

		global_out, global_nets = self.global_features_network(low_out_scaled, name="global_features_network")

		fused = self.fuse_net(global_out, middle_out, name="fusenet")
		
		self.colorization_out, self.colorization_nets = self.colorization_network(fused, name="colorization_network")

		self.trainable_vars = tf.trainable_variables()
		
	def build_loss(self):

		pred_ab = self.colorization_out
		
		#change original rgb image to cielab color space
		gt_lab = rgb_to_lab(self.orig)
		L_chan, a_chan, b_chan = preprocess_lab(gt_lab)
		self.gt_lab = tf.stack([L_chan, a_chan, b_chan], axis=3)
		gt_ab = tf.stack([a_chan, b_chan], axis=3)
		
		self.loss = tf.reduce_mean(tf.square(pred_ab - gt_ab))

		#reconstruct rgb image using pred ab
		pred_a, pred_b = tf.unstack(pred_ab, axis=3)
		self.pred_lab = deprocess_lab(L_chan, pred_a, pred_b)

		self.pred_rgb = deprocess(lab_to_rgb(self.pred_lab))



	def low_level_features_network(self, input, reuse=False, name="low_network"):
		nets = []
		with tf.variable_scope(name, reuse=reuse) as scope:

			conv1 = tf.contrib.layers.conv2d(input, 32,
											 kernel_size=3, stride=2,
											 padding="VALID",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv1")
			conv1 = batch_norm(conv1, name="conv1_bn")
			nets.append(conv1)

			conv2 = tf.contrib.layers.conv2d(conv1, 64,
											 kernel_size=3, stride=1,
											 padding="SAME",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv2")
			conv2 = batch_norm(conv2, name="conv2_bn")
			nets.append(conv2)

			conv3 = tf.contrib.layers.conv2d(conv2, 64,
											 kernel_size=3, stride=2,
											 padding="VALID",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv3")
			conv3 = batch_norm(conv3, name="conv3_bn")
			nets.append(conv3)

			conv4 = tf.contrib.layers.conv2d(conv3, 64,
											 kernel_size=3, stride=1,
											 padding="SAME",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv4")
			conv4 = batch_norm(conv4, name="conv4_bn")
			nets.append(conv4)

			conv5 = tf.contrib.layers.conv2d(conv4, 128,
											 kernel_size=3, stride=2,
											 padding="VALID",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv5")
			conv5 = batch_norm(conv5, name="conv5_bn")
			nets.append(conv5)

			conv6 = tf.contrib.layers.conv2d(conv5, 256,
											 kernel_size=3, stride=1,
											 padding="SAME",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv6")	
			conv6 = batch_norm(conv6, name="conv6_bn")
			nets.append(conv6)

			output = conv6


			return output, nets								 			

	def global_features_network(self, input, name="global_network"):
		nets = []
		with tf.variable_scope(name) as scope:
			conv1 = tf.contrib.layers.conv2d(input, 256,
											 kernel_size=3, stride=2,
											 padding="VALID",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv1")
			conv1 = batch_norm(conv1, name="conv1_bn")	
			nets.append(conv1)

			conv2 = tf.contrib.layers.conv2d(conv1, 256,
											 kernel_size=3, stride=1,
											 padding="SAME",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv2")	
			conv2 = batch_norm(conv2, name="conv2_bn")
			nets.append(conv2)

			conv3 = tf.contrib.layers.conv2d(conv2, 256,
											 kernel_size=3, stride=2,
											 padding="VALID",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv3")
			conv3 = batch_norm(conv3, name="conv3_bn")
			nets.append(conv3)

			conv4 = tf.contrib.layers.conv2d(conv3, 256,
											 kernel_size=3, stride=1,
											 padding="SAME",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv4")
			conv4 = batch_norm(conv4, name="conv4_bn")
			nets.append(conv4)

			flattened = tf.contrib.layers.flatten(conv4)
			fc1 = tf.contrib.layers.fully_connected(flattened, num_outputs=512,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=tf.nn.relu,
													scope="fc1"
													)
			fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=256,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=tf.nn.relu,
													scope="fc2"
													)
			fc3 = tf.contrib.layers.fully_connected(fc2, num_outputs=128,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=tf.nn.relu,
													scope="fc3"
													)
			output = fc3

			return output, nets

	def middle_level_features_network(self, input, name="middle_network"):
		nets = []
		with tf.variable_scope(name) as scope:
			conv1 = tf.contrib.layers.conv2d(input, 356,
											 kernel_size=3, stride=1,
											 padding="SAME",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv1")
			conv1 = batch_norm(conv1, name="conv1_bn")
			nets.append(conv1)

			conv2 = tf.contrib.layers.conv2d(conv1, 128,
											 kernel_size=3, stride=1,
											 padding="SAME",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,											 
											 scope="conv2")
			conv2 = batch_norm(conv2, name="conv2_bn")
			nets.append(conv2)

			output = conv2


			return output, nets



	def colorization_network(self, input, name="colorization_network"):
		nets = []
		with tf.variable_scope(name) as scope:
			deconv1 = tf.contrib.layers.conv2d_transpose(input, 128,
														 kernel_size=3, stride=1,
														 padding="SAME",
														 weights_initializer=tf.contrib.layers.xavier_initializer(),
														 activation_fn=tf.nn.relu,
														 scope="deconv1")
			deconv1 = batch_norm(deconv1, name="deconv1_bn")
			nets.append(deconv1)

			deconv2 = tf.contrib.layers.conv2d_transpose(deconv1, 64,
														 kernel_size=3, stride=2,
														 padding="VALID",
														 weights_initializer=tf.contrib.layers.xavier_initializer(),
														 activation_fn=tf.nn.relu,
														 scope="deconv2")	
			deconv2 = batch_norm(deconv2, name="deconv2_bn")		
			nets.append(deconv2)

			deconv3 = tf.contrib.layers.conv2d_transpose(deconv2, 64,
														 kernel_size=3, stride=1,
														 padding="SAME",
														 weights_initializer=tf.contrib.layers.xavier_initializer(),
														 activation_fn=tf.nn.relu,
														 scope="deconv3")
			deconv3 = batch_norm(deconv3, name="deconv3_bn")		
			nets.append(deconv3)

			deconv4 = tf.contrib.layers.conv2d_transpose(deconv3, 32,
														 kernel_size=3, stride=2,
														 padding="VALID",
														 weights_initializer=tf.contrib.layers.xavier_initializer(),
														 activation_fn=tf.nn.relu,
														 scope="deconv4")
			deconv4 = batch_norm(deconv4, name="deconv4_bn")	
			nets.append(deconv4)			

			deconv5 = tf.contrib.layers.conv2d_transpose(deconv4, 32,
														 kernel_size=3, stride=2,
														 padding="VALID",
														 weights_initializer=tf.contrib.layers.xavier_initializer(),
														 activation_fn=tf.nn.relu,
														 scope="deconv5")
			deconv5 = batch_norm(deconv5, name="deconv5_bn")
			nets.append(deconv5)		

			deconv6 = tf.contrib.layers.conv2d_transpose(deconv5, 2,
														 kernel_size=3, stride=1,
														 padding="SAME",
														 weights_initializer=tf.contrib.layers.xavier_initializer(),
														 activation_fn=tf.nn.tanh,
														 scope="deconv6")			
			nets.append(deconv6)

			output = deconv6

			return output, nets
			#Continue workign on this part


	def fuse_net(self, global_input, middle_input, name="fusenet"):
		with tf.variable_scope(name) as scope:
			middle_shape = middle_input.get_shape()
			reshaped_global = global_input[:,tf.newaxis,:]
			reshaped_global = tf.tile(reshaped_global, [1, middle_shape[1].value*middle_shape[2].value, 1])
			reshaped_global = tf.reshape(reshaped_global, shape=[middle_shape[0].value, middle_shape[1].value, middle_shape[2].value, -1])

			fuse_ready = tf.concat([middle_input, reshaped_global], axis=3)
			fuse_ready_shape = fuse_ready.get_shape()
			fuse_ready = fuse_ready[...,tf.newaxis]
			fuse_weight = tf.get_variable("weight", shape=[fuse_ready_shape[0], fuse_ready_shape[1], fuse_ready_shape[2], 256, 128])		
			fused = tf.matmul(fuse_ready, fuse_weight, transpose_a=True)
			fused = tf.squeeze(fused, axis=-2)

			return fused			








