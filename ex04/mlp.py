#
# TU Berlin - Machine Intelligence I WS11/12 - Exercise 4
# A Simple MLP
# authors: Rolf Schroeder & Robin Vobruba
#

import math, random, time

random.seed()
last_rand=random.random()

# Define the some vars

# interval (from -alpha to alpha)
alpha = 0.5
# number of neurons per layer (counting the bias)
# n[layer] = no.
n = [2,3,1]
# weights
# w[depth][neuronfrom][neuronto]
w = []
# activities
# s[depth][neuronfrom]
s = []
# input data set
inputs = []
# output data set
outputs = []

def transfer_func_hidden(x):
	return math.tanh(x)
def transfer_func_output(x):
	return x
def error_func(y, yt):
	return (y-yt)**2
def matrix(i, j, default=0.0):
	m = []
	for k in range(i):
		m.append([default]*j)
	return m
def random_in_interval(x):
	return  random.uniform(-x,x)


###### INIT ######
# weigths
for i in range(len(n)-1):
	w.append([])
	w[i] = matrix(n[i], n[i+1])
# activities
for i in range(len(n)):
	s.append([])
	s[i] = [0.0]*n[i]

# populate weights
for i in range(len(w)):
	for j in range(len(w[i])):
		for k in range(len(w[i][j])):
			w[i][j][k] = random_in_interval(alpha)


###### LEARNING ######
# Read the data file
bias = 1
try:
	data_f = open('data.txt', 'r')
	line = data_f.readline()
	while line != "":
		parts = line.split()
		# read the x (input for the sample), and prepend it
		# with the bias (constant threshold multiplier)
		inputs.append([bias, float(parts[0])])
		# read the y (output for the sample)
		outputs.append([float(parts[1])])
		line = data_f.readline()
except:
	print "Failed to parse the data file!"
	exit(-1)
finally:
	data_f.close()



for training in range(len(inputs)):
	x = inputs[training]
	# "activities" of first layer (e.g. inputs)
	for i in range(len(x)):
		s[0][i] = x[i]

	# activities for hidden and layers
	for current_layer in range(1, len(n)): # for each layer
		previous_layer = current_layer - 1
		for current_neuron in range(n[current_layer]): # for each neuron in this layer
			h = 0.0 # total input for this neuron
			for previous_neuron in range(len(s[previous_layer])): # for each neuron in the previous layer
				h = h + w[previous_layer][previous_neuron][current_neuron] * s[previous_layer][previous_neuron]
			if (current_layer != len(n)-1):
				f = transfer_func_hidden
			else:
				f = transfer_func_output
			s[current_layer][current_neuron] = f(h)


	print s
