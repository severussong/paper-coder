# -*- coding: utf-8 -*-
#/usr/bin/python
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('/root/github')
import tensorflow as tf
from tensorflow.python.ops import variables
from hyperparams import Hyperparams as hp
from data_load import load_train_data_test, load_en_vocab
import numpy as np
from modules import *
import os, codecs
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class Graph():
    def __init__(self, is_training=False):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen),name='source')
            self.y = tf.placeholder(tf.int32, shape=(None, 1),name='target')

            # Load vocabulary
            en2idx, idx2en = load_en_vocab()

            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc = embedding(self.inputs,
                                      vocab_size=len(en2idx),
                                      num_units=hp.hidden_units,
                                      scale=True,
                                      scope="enc_embed")
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
			self.enc_1 = tf.expand_dims(concatnate_power(self.enc_1),axis=1)
            self.enc_2 = tf.expand_dims(concatnate_power(self.enc_2),axis=1)
	        self.enc = tf.concat([self.enc_1,self.enc_2],axis=1)

            self.enc = tf.expand_dims(self.enc,-1)
			
			self.enc = tf.nn.max_pool(self.enc,[1,hp.maxlen,1,1],[1,1,1,1],'VALID')
			
			self.enc = tf.squeeze(self.enc,[1,3])

			self.enc = tf.layers.dense(self.enc, 5 ,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.tanh) 
            self.enc = tf.nn.softmax(self.enc)
            self.pred_result = tf.argmax(self.enc,axis=1)
            self.label = tf.squeeze(self.y,1)	
			self.label = self.label - 1
	        
 




if __name__ == '__main__':
	
	_,vocab_size = load_en_vocab()
	del _
	gc.collect()
	g = Graph(vocab_size)
	num_batch=len(X) // batch_size
    preds = []
    labels = []
	gs = 0
    with tf.Session(graph=g.graph) as sess:
            g.init.run()
			saver = tf.train.Saver()
            saver.restore(sess, 'location of model')
            X,Y = load_train_data_test(hp.test_data)
			for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                gs = gs+1
                if gs%1000 ==0:
                    print(gs)
                x = X[step * batch_size: (step + 1) * batch_size]
                y = Y[step * batch_size: (step + 1) * batch_size]
                result,label_temp = sess.run(g.pred_result,g.labels,feed_dict={g.x:x , g.y:y})
                for i in range(len(result)):
                    preds.append(result[i])
                    labels.append(label_temp[i])

    accuracy = accuracy_score(labels,preds)
    print('accuracy:'+str(accuracy))

