#
# TU Berlin - Machine Intelligence I WS11/12 - Exercise 4
# A Simple MLP
# authors: Rolf Schroeder & Robin Vobruba
#

import math

# Define the some vars

# number of neurons per layer
# n[layer] = no.
n = [2,3,1]
# weights
# w[layerfrom][neuronfrom][layerto][neuronto]
w = []
# activities
# s[layer][neuron]
s = []
# transfer function
# t[layer][neuron]
t = []
# input data set (includes the bias at x[0])
x = []
# output data set
y = []
# interval (from -alpha to alpha)
alpha = 0.5

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
	return random.uniform(-x,x)

# Read the data file
try:
	data_f = open('data.txt', 'r')
	line = data_f.readline()
	while line != "":
		parts = line.split()
		# read the x (input for the sample), and prepend it
		# with the bias (constant threshold multiplier)
		x.append([1, float(parts[0])])
		# read the y (output for the sample)
		y.append([float(parts[1])])
		line = data_f.readline()
except:
	print "Failed to parse the data file!"
	exit(-1)
finally:
	data_f.close()

for i in range(0,len(n)-2):
	w.append([])
	w[i] = matrix(n[i], n[i+1], random_in_interval(alpha))

# Initialize the weights
for i in range(0, len(n) - 2): # for each layer except the last one
	w.append([])
	w[i].append([])
	for j in range(0, n[i] - 1): # for each neuron in that layer
		w[i][j].append([])
		for k in range(0, n[i+1] - 1): # for each neuron from the next layer
			w[i][j][k].append([])

			w[i][j][i+1][k] = rand(alpha)
			print "ie"
#	w[i]
			

