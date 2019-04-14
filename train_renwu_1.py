# -*- coding: utf-8 -*-
#/usr/bin/python

from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import variables
from hyperparams import Hyperparams as hp
from data_load import load_en_vocab, load_train_data
from modules import *
import os, codecs
import random
from tqdm import tqdm
import gc

class Graph():
	def __init__(self,vocab_size ,is_training=True):
		self.graph = tf.Graph()
		self.vocab_size = vocab_size
		with self.graph.as_default():
			self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen),name='source')
			self.y = tf.placeholder(tf.int32, shape=(None, 1),name='target')

			with tf.variable_scope("encoder"):
                ## Embedding
				self.enc = embedding(self.x,
                                      vocab_size=self.vocab_size,
                                      num_units=hp.hidden_units,
                                      scale=True,
                                      scope="enc_embed")

                ## Positional Encoding
				if hp.sinusoid:
					self.enc += positional_encoding(self.x,
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="enc_pe")
				else:
					self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.maxlen,
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="enc_pe")


                ## Dropout
				self.enc = tf.layers.dropout(self.enc,
                                            rate=hp.dropout_rate,
                                            training=tf.convert_to_tensor(is_training))

                ## Blocks
			    with tf.variable_scope("num_blocks_1"):
                        ### Multihead Attention
					self.enc_1 = multihead_attention(queries=self.enc,
                                                        keys=self.enc,
                                                        num_units=hp.hidden_units,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)

                        ### Feed Forward
					self.enc_1 = facebook(self.enc_1, num_units=[hp.hidden_units])
	            with tf.variable_scope("num_blocks_2"):
                   	self.enc_2 = multihead_attention(queries=self.enc_1,
                                                        keys=self.enc_1,
                                                        num_units=hp.hidden_units,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                    self.enc_2 = facebook(self.enc_2, num_units=[hp.hidden_units])	

			self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, hp.hidden_units],
						                                                              stddev=1.0/math.sqrt(hp.hidden_units)))
			self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
			
			self.enc_temp = tf.expand_dims(self.enc_2,-1)
			
			self.enc_temp = tf.nn.max_pool(self.enc_temp,[1,hp.maxlen,1,1],[1,1,1,1],'VALID')
			
			self.enc_temp = tf.squeeze(self.enc_temp,[1,3])

			self.loss = tf.reduce_mean(
					               tf.nn.nce_loss(
								   weights = self.nce_weight,  
								   biases = self.nce_biases,   
								   labels = self.y, 
								   inputs = self.enc_temp,            
				           		   num_sampled = 5, 
					               num_classes = self.vocab_size))

			if is_training:
				self.global_step = tf.Variable(0, name='global_step', trainable=False)
				self.optimizer = tf.train.AdagradOptimizer(learning_rate=hp.lr)
				self.grads_ = tf.gradients(self.loss, tf.trainable_variables())
				self.grads, _ = tf.clip_by_global_norm(self.grads_, hp.grad_norm)
				self.grads_and_vars = list(zip(self.grads, tf.trainable_variables()))
				self.train_op = self.optimizer.apply_gradients(grads_and_vars=self.grads_and_vars,global_step = self.global_step)
				self.saver = tf.train.Saver(var_list=[var for var in variables._all_saveable_objects()],max_to_keep=10)
				self.init = tf.initialize_all_variables()
				  
if __name__ == '__main__':
	_,vocab_size = load_en_vocab()
	del _
	gc.collect()
	g = Graph(vocab_size)
	gs = 0
	with tf.Session(graph=g.graph) as sess:
		g.init.run()
		for epoch in range(1, hp.num_epochs+1):
			for file_name in os.listdir(hp.source_train):
				X,Y = load_train_data(file_name)
				num_batch = len(X) // hp.batch_size
				rank = list(range(num_batch))
				random.shuffle(rank)
				for step in tqdm( rank , total=num_batch, ncols=70, leave=False, unit='b'):
					gs = gs+1
					x = X[step * hp.batch_size: (step + 1) * hp.batch_size]
					y = Y[step * hp.batch_size: (step + 1) * hp.batch_size]
					_,loss = sess.run([g.train_op,g.loss],feed_dict={g.x:x , g.y:y})
					if gs % 5000 == 0:
						print("step"+str(gs)+':'+str(loss))
						g.saver.save(sess,hp.modeldir+'/model_global_step_%d' % gs)
