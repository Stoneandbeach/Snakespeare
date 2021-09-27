# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:27:25 2021

@author: Sten Åstrand

Simple, hand-coded RNN for text synthesis.

"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt
import urllib

path = r'C:\Users\Stoneandbeach\Dropbox\Plugg\VT21\Master\python\RNN test'
filename = r'bibel.txt'

# This example uses Shakespeare's sonnets as the basis for the training
file = urllib.request.urlopen('https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt')
b = file.read()
raw_text = b.decode('utf-8').replace('\\n', '\n')
file.close()

training_text = raw_text
seed = '' # A custom seed text can be used to always start the sampling from a given context

training_length = len(training_text)
chars = list(set(training_text))

ix_to_char = {i:char for i, char in enumerate(chars)}
char_to_ix = {char:i for i, char in enumerate(chars)}

#Initialize
lr = 1e-1 # Learning rate
wd = 1e-3 # Weight decay
hidden_size = 150

vocab_size = len(chars)
h = np.zeros([hidden_size, 1])
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros([hidden_size, 1])
by = np.zeros([vocab_size, 1])

#%% Definitions

def update(x, h, Wxh, Whh, bh):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh) # Tanh activation of input and previous hidden state
    return h

def predict(h, Why, by):
    y = np.dot(Why, h) + by # Prediction from (updated) hidden state
    return y

def ix_vec_to_char(vec):
    return ix_to_char[int(np.where(vec == vec.max())[0])]

def learn(y, t, x, h, hprev):
    g = h**2
    p = np.exp(y) / np.sum(np.exp(y))
    dy = p - t
    loss = np.linalg.norm(dy)**2
    WhyTdy = np.dot(Why.T, dy)
    WhyTdyg = WhyTdy * (1 - g)
    dWxh = np.dot(WhyTdyg, x.T)
    dWhh = np.dot(WhyTdyg, hprev.T)
    dWhy = np.dot(dy, h.T)
    dbh = WhyTdyg
    dby = dy
    
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -1, 1, out=dparam)
        
    return dWxh, dWhh, dWhy, dby, dbh, loss   

def step(h):
    h = update(x, h, Wxh, Whh, bh)
    y = predict(h, Why, by)
    return h, y

def sample(seed, length, h = None):
    if h is None:
        h = np.zeros(bh.shape)
    for char in seed:
        x = np.zeros([vocab_size, 1])
        x[char_to_ix[char]] = 1
        h = update(x, h, Wxh, Whh, bh)
    output = ''
    for i in range(length):
        y = predict(h, Why, by)
        
        p = np.exp(y) / np.sum(np.exp(y))
        
        idx = np.random.choice(range(vocab_size), p = p.ravel())
        
        x = np.zeros_like(y)
        x[idx] = 1
        h = update(x, h, Wxh, Whh, bh)
        next_char = ix_to_char[idx]
        output += next_char
    return output


# Setting up input at target data
xs = np.zeros([vocab_size, training_length - 1])
ts = np.zeros([vocab_size, training_length - 1])
for i in range(training_length - 1):
    xs[char_to_ix[training_text[i]], i] = 1
    ts[char_to_ix[training_text[i+1]], i] = 1

#%% Training run
epochs = 30
epoch = 0

start_time = time.perf_counter()

mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mby, mbh = np.zeros_like(by), np.zeros_like(bh)

while epoch < epochs:
    
    smooth_loss = 0    
    h = np.zeros([hidden_size, 1]) #Reset hidden state for next epoch
    epoch += 1
    for i in range(training_length - 1):
        x = xs[:, i].reshape([vocab_size, 1]) # Current input...
        t = ts[:, i].reshape([vocab_size, 1]) # ...and target
        hprev = h
        h, y = step(h)
        dWxh, dWhh, dWhy, dby, dbh, loss = learn(y, t, x, h, hprev)
        smooth_loss += loss
        
        for mem, param, dparam in zip([mWxh, mWhh, mWhy, mby, mbh],
                                      [Wxh, Whh, Why, by, bh],
                                      [dWxh, dWhh, dWhy, dby, dbh]):
            mem += dparam ** 2
            param += -lr * dparam * (1-wd) / np.sqrt(mem + 1e-8) #adagrad update
        
    print('_______________')
    print('Epoch number', epoch)
    print('Loss', smooth_loss / training_length)
    print()
    print('Sampling...\n')
    print(seed + sample(seed, 200, h=h))
    print()

print('Runtime was', time.perf_counter() - start_time, 'seconds.')