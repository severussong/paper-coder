# -*- coding: utf-8 -*-
#/usr/bin/python

from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import re
import codecs
import regex

def load_en_vocab():
	vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
	word2idx = {word: idx for idx, word in enumerate(vocab)}
	return word2idx,len(word2idx)

def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data_2(source_sents, target_sents):
    word2idx = load_en_vocab()

    # Index
    sentence, target = [], []
    for source_sent, target_sent  in zip(source_sents, target_sents):
        x = [word2idx.get(word, 1) for word in (source_sent + u" </S>").split()]
        y = [word2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <=hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))

    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    return X, Y

def create_data_3(source_sents, target_sents):
    en2idx, idx2en = load_en_vocab()
    de2idx, idx2de = load_de_vocab()
    # Index
    x_list, y_list = [], []
    for source_sent, target_sent  in zip(source_sents, target_sents):
        x = [en2idx.get(word, 1) for word in (source_sent + u" </S>").split()]
        y = [de2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <=hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))

    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    return X, Y

def create_data_word2vec(en_sents):
	word2idx,_ = load_en_vocab()

    # Index
	x_list, y_list = [], []
	for source_sent in en_sents:
		data = source_sent.split(' ##')
		x = [word2idx.get(word, 1) for word in (data[0] + u" </S>").split()]
		y = [word2idx.get(data[1], 1)]
		if len(x) <=hp.maxlen:
			x_list.append(np.array(x))
			y_list.append(np.array(y))

	X = np.zeros([len(x_list), hp.maxlen], np.int32)
	Y = np.asarray(y_list, np.int32)
	for i, x in enumerate(x_list):
		X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
	return X, Y

def create_data_second(en_sents):
	word2idx,_ = load_en_vocab()

    # Index
	x_list, y_list = [], []
	for source_sent in en_sents:
		data = source_sent.split(' ##')
		x = [word2idx.get(word, 1) for word in (data[0] + u" </S>").split()]
		try:
			y = [int(float(data[1]))]
		except Exception as e:
			continue
		if y[0] not in [1,2,3,4,5]:
			continue
		if len(x) <=hp.maxlen:
			x_list.append(np.array(x))
			y_list.append(np.array(y))
		else:
			x_list.append(np.array(x[0:hp.maxlen]))
			y_list.append(np.array(y))

	X = np.zeros([len(x_list), hp.maxlen], np.int32)
	Y = np.asarray(y_list, np.int32)
	for i, x in enumerate(x_list):
		X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
	return X, Y


def load_train_data_renwu1(files):
	en_sents = [line for line in codecs.open(hp.source_train+files, 'r', 'utf-8').read().split("\n") if line]
	X, Y= create_data_word2vec(en_sents)
	return X, Y

def load_train_data_predict(files):
	en_sents = [line for line in codecs.open(hp.source_train+files, 'r', 'utf-8').read().split("\n") if line]
	x, y= create_data_second(en_sents)
	return x, y

def load_train_data_next_sentence(files):
	en_sents = [line for line in codecs.open(hp.source_train+files, 'r', 'utf-8').read().split("\n") if line]
	x, y= create_data_2(en_sents)
	return x, y

def load_train_data_translate(files):
	en_sents = [line for line in codecs.open(hp.source_train+files, 'r', 'utf-8').read().split("\n") if line]
	x, y= create_data_3(en_sents)
	return x, y

def load_train_data(files):
	en_sents = [line for line in codecs.open(hp.source_train+files, 'r', 'utf-8').read().split("\n") if line]
	X, Y= create_data_word2vec(en_sents)
	return X, Y

def load_train_data_2(files):
	en_sents = [line for line in codecs.open(hp.source_train+files, 'r', 'utf-8').read().split("\n") if line]
	X, Y= create_data_second(en_sents)
	return X, Y

def load_train_data_test(files):
    en_sents = [line for line in codecs.open(files, 'r', 'utf-8').read().split("\n") if line]
    X, Y= create_data_second(en_sents)
    return X, Y

