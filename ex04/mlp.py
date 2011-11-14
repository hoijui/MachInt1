#
# TU Berlin - Machine Intelligence I WS11/12 - Exercise 4
# A Simple MLP
# authors: Rolf Schroeder & Robin Vobruba
#

import math, random, time

random.seed()
last_rand=random.random()


# Define the networks component vars

# interval (from -alpha to alpha)
alpha = 0.5

learnRate = 0.5

# number of neurons per layer (counting the bias)
# n[layer] = no.
n = [1, 3, 1]

# weights
# w[layerId][neuronFrom][neuronTo]
# neuronFrom == 0 -> the bias weight
w = []

# local inputs
h = []

# transfer functions
t = []
td = []

# local errors
d = []

# gradients
grad = []

# activities
# s[depth][neuronfrom]
S = []

# input data set
inputs = []

# output data set
outputs = []



def transfer_hidden(x):
	return math.tanh(x)

def transfer_hidden_deriv(x):
	return 1 - math.tanh(x)**2

def transfer_output(x):
	return x

def transfer_output_deriv(x):
	return 1

def weightInitializer():
	return random_in_interval(alpha)

def initZero():
	return 0.0

def error(y, yt):
	return 0.5 * ((y - yt)**2)

def error_deriv(y, yt):
	return (y - yt)

def matrix(columns, rows, initializerFunc=initZero):
	m = [[0 for r in range(rows)] for c in range(columns)]
	for c in range(columns):
		for r in range(rows):
			m[c][r] = initializerFunc()
	return m

def random_in_interval(x):
	return  random.uniform(-x, x)


###### INIT ######
# weigths
for layerId in range(len(n) - 1):
	w.append(matrix(n[layerId] + 1, n[layerId + 1], weightInitializer))
	h.append([0.0] * n[layerId + 1])
	d.append([0.0] * n[layerId + 1])
	grad.append([0.0] * n[layerId + 1])

# transfer functions
t.append([transfer_hidden] * n[1])
t.append([transfer_output] * n[2])

# transfer functions
td.append([transfer_hidden_deriv] * n[1])
td.append([transfer_output_deriv] * n[2])

# activities
for layerId in range(len(n)):
	S.append([0.0] * n[layerId])


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
		inputs.append([float(parts[0])])
		# read the y (output for the sample)
		outputs.append([float(parts[1])])
		line = data_f.readline()
except:
	print "Failed to parse the data file!"
	exit(-1)
finally:
	data_f.close()



# Forward propagation
def forwardPropLayer(layerId):
	for neuronId in range(n[layerId]):
		nPre = n[layerId - 1]
		# bias * thresholdWeight
		hCur = 1 * w[layerId - 1][0][neuronId]
		for preNeuronId in range(n[layerId - 1] + 1): 
			biasPlusS = [1.0] + S[layerId - 1]
			Spre = biasPlusS[preNeuronId]
			wcur = w[layerId - 1][preNeuronId][neuronId]
			hCur = hCur + (Spre * wcur)
		h[layerId - 1][neuronId] = hCur
		S[layerId][neuronId] = t[layerId - 1][neuronId](hCur)



def forwardPropStep(dataIndex):
	x = inputs[dataIndex][0]
	# "activities" of layer 0 (e.g. inputs)
	S[0] = [x]

	# activities for hidden layers
	for layerId in range(len(n) - 1):
		forwardPropLayer(layerId + 1)

	y_T = S[len(n) - 1][0]
	y = outputs[dataIndex][0]
	e = error(y, y_T)

	print x, "\t", y, "\t", y_T, "\t", e
	return e


# Backwards propagation
def backwardPropLayer(layerId):
	for neuronId in range(n[layerId] + 1):
		wdSum = 0.0
		for postNeuronId in range(n[layerId - 1]):
			print "layerId: ", layerId, "neuronId: ", neuronId, "postNeuronId: ", postNeuronId
			print w[layerId][neuronId][postNeuronId]
			print d[layerId][postNeuronId]
			wdSum += w[layerId][neuronId][postNeuronId] * d[layerId + 1][postNeuronId]
		d[layerId][neuronId] = td[layerId][neuronId](h[layerId][neuronId]) * wdSum
		grad[][][] = 
#learnRate


def backwardPropStep(dataIndex, e):
	# local errors of the output neurons
	d[1] = [td[1][0](h[1][0])]

	# activities for hidden layers
	lli = range(len(n) - 2)
	lli.reverse()
	for layerId in lli:
		backwardPropLayer(layerId)


print  "x\ty\ty_T\terror"
for dataIndex in range(len(inputs)):
	e = forwardPropStep(dataIndex)
	backwardPropStep(dataIndex, e)
