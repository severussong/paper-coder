# -*- coding: utf-8 -*-
#/usr/bin/python

class Hyperparams:

	source_train = 'corpora/amazon_train/'
	target_train = 'corpora/amazon_train/'
	test_data = 'corpora/test_data'

    # training
	batch_size = 32# alias = N
	lr = 0.001 # learning rate. In paper, learning rate is adjusted to the global step.
	logdir = '/word2vec/log' # log directory
	modeldir = '/root/pan/final_con_predict/amazon'
	checkpoint = 1000
	grad_norm = 5

    # model
	maxlen = 20 # Maximum number of words in a sentence. alias = T.
	min_cnt = 15 # words whose occurred less than min_cnt are encoded as <UNK>.
	hidden_units = 256 # alias = C
	num_blocks = 1 # number of encoder/decoder blocks
	num_epochs = 2
	num_heads = 8
	dropout_rate = 0.1
	sinusoid = True # If True, use sinusoid. If false, positional embedding.
