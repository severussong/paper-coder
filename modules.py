# -*- coding: utf-8 -*-
#/usr/bin/python

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math

def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
		inputs_shape = inputs.get_shape()
		params_shape = inputs_shape[-1:]
		mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
		beta= tf.Variable(tf.zeros(params_shape))
		gamma = tf.Variable(tf.ones(params_shape))
		normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
		outputs = gamma * normalized + beta
	return outputs

def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):

	with tf.variable_scope(scope, reuse=reuse):
		lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0,
										   stddev=1.0/math.sqrt(num_units)))
		if zero_pad:
			lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)

		outputs = tf.nn.embedding_lookup(lookup_table, inputs)

		if scale:
			outputs = outputs * (num_units ** 0.5)

	return outputs


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):

	N = tf.shape(inputs)[0]
	T = inputs.get_shape().as_list()[1]
	with tf.variable_scope(scope, reuse=reuse):
		position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
		position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
		position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
		position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
		lookup_table = tf.convert_to_tensor(position_enc,tf.float32)

		if zero_pad:
			lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
		outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

		if scale:
			outputs = outputs * num_units**0.5

		return outputs



def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
	with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
		if num_units is None:
			num_units = queries.get_shape().as_list[-1]

        # Linear projections
		Q = tf.layers.dense(queries, num_units,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu) # (N, T_q, C)
		K = tf.layers.dense(keys, num_units,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu) # (N, T_k, C)
		V = tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu) # (N, T_k, C)

        # Split and concat
		Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
		K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
		V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        # Multiplication
		outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

        # Scale
		outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
		key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
		key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
		key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

		paddings = tf.ones_like(outputs)*(-2**32+1)
		outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
		if causality:
			diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
			tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
			masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

			paddings = tf.ones_like(masks)*(-2**32+1)
			outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
		outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        # Query Masking
		query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
		query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
		query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
		outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
		outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
		outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
		outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        # Residual connection
		outputs += keys

        # Normalize
		outputs = normalize(outputs) # (N, T_q, C)

	return outputs

def feedforward(inputs,
                num_units=[256],
                scope="multihead_attention",
                reuse=None):

	with tf.variable_scope(scope, reuse=reuse):
        # Readout layer
		params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,"kernel_initializer":tf.contrib.layers.xavier_initializer(),
                  "activation": None, "use_bias": True}
		outputs = tf.layers.conv1d(**params)

        # Residual connection
		outputs += inputs
        # Normaliz
		outputs = normalize(outputs)

	return outputs

def feedforward_conv(inputs,
                     cnn_layers=2,
                     kernel_size=[3,3],
                     scope="multihead_attention",
                     reuse=None):

    with tf.variable_scope(scope,reuse=reuse):
        in_dim = int(inputs.get_shape()[-1])
        out_dim = in_dim
        next_layers = inputs
        for i in range(cnn_layers):
            with tf.variable_scope('cnn_layer_'+str(i), reuse=reuse):
                 res_layer = inputs
                 V = tf.get_variable('V', shape=[kernel_size[i], in_dim, out_dim],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(mean=0,
                                      stddev=tf.sqrt(4.0/(kernel_size[i]*in_dim))),
                                      trainable=True)

                 V_norm = tf.norm(V.initialized_value(), axis=[0,1])
                 g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
                 b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
                 W = tf.reshape(g, [1,1,out_dim])*tf.nn.l2_normalize(V,[0,1])
                 inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)
                 inputs = (inputs + res_layer)*tf.sqrt(0.5)
        # Residual connection
        outputs = (next_layers + inputs)*tf.sqrt(0.5)
        outputs = normalize(outputs)
    return outputs

def feedforward_dnn(inputs,
                    dnn_layers=3,
                    scope="multihead_attention",
                    reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        out_dim = int(inputs.get_shape()[-1])
        for i in range(dnn_layers):
            with tf.variable_scope('dnn_layer_'+str(i), reuse=reuse):
                inputs = tf.layers.dense(inputs, out_dim, kernel_initializer=tf.contrib.layers.xavier_initializer() ,activation=tf.nn.tanh)

    return inputs

def label_smoothing(inputs, epsilon=0.1):
	
	K = inputs.get_shape().as_list()[-1] # number of channels
	return ((1-epsilon) * inputs) + (epsilon / K)

def facebook(inputs,
                num_units=[256*2],
                scope="multihead_attention",
                reuse=None):

	with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
		params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,"kernel_initializer":tf.contrib.layers.xavier_initializer(),
                  "activation": tf.nn.relu, "use_bias": True}
		outputs = tf.layers.conv1d(**params)
		outputs = tf.multiply(outputs[:,:,0:num_units[0]/2],tf.sigmoid(outputs[:,:,num_units[0]/2:num_units[0]]))

        # Residual connection
		outputs += inputs
        # Normaliz
		outputs = normalize(outputs)

	return outputs

def gen_power(inputs,p):
	inputs = tf.pow(inputs,p)
	inputs = tf.reduce_mean(inputs,1)
	power = 1.0/float(p)
	result = tf.pow(inputs,power)
	return result	

def concatcate_power(inputs):
    enc_1 = tf.expand_dims(inputs,-1)
    enc_1 = tf.nn.max_pool(enc_1,[1,hp.maxlen,1,1],[1,1,1,1],'VALID')
	enc_1 = tf.squeeze(enc_1,[1,3])
    enc_1 = tf.layers.dense(enc_1, 80 ,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu) 
    enc_2 = tf.reduce_mean(self.enc,1)
    enc_2 =  tf.layers.dense(enc_2, 80 ,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu) 
    enc_3 = gen_power(self.enc,2)
    enc_3 =  tf.layers.dense(enc_3, 80 ,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
    return tf.concat([enc_1,enc_2,enc_3],axis=1)	
