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

biasS = 1.0

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

def toApproximate(x):
	return math.sin(2*math.pi*x)


def matrix(columns, rows, initializerFunc=initZero):
	m = [[0 for r in range(rows)] for c in range(columns)]
	for c in range(columns):
		for r in range(rows):
			m[c][r] = initializerFunc()
	return m

def random_in_interval(x):
	return random.uniform(-x, x)

def deciRange(start, stop, step):
	rng = []
	cur = start
	while cur <= stop:
		rng.append(cur)
		cur += step
	return rng


###### INIT ######
# weigths
for layerId in range(len(n) - 1):
	w.append(   matrix(n[layerId] + 1, n[layerId + 1], weightInitializer))
	grad.append(matrix(n[layerId] + 1, n[layerId + 1], initZero))
	h.append([0.0] * n[layerId + 1])
	d.append([0.0] * n[layerId + 1])

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
			SWithBias = [biasS] + S[layerId - 1]
			Spre = SWithBias[preNeuronId]
			wcur = w[layerId - 1][preNeuronId][neuronId]
			hCur = hCur + (Spre * wcur)
		h[layerId - 1][neuronId] = hCur
		S[layerId][neuronId] = t[layerId - 1][neuronId](hCur)



def forwardPropStep(x):
	# "activities" of layer 0 (e.g. inputs)
	S[0] = [x]

	# activities for hidden layers
	for layerId in range(len(n) - 1):
		forwardPropLayer(layerId + 1)

	y_T = S[len(n) - 1][0]

	return y_T


# Backwards propagation
def backwardPropLayerLocalErrors(layerId):
	for neuronId in range(n[layerId] + 1):
		wdSum = 0.0
		for postNeuronId in range(n[layerId + 1]):
			#print "layerId: ", layerId, "neuronId: ", neuronId, "postNeuronId: ", postNeuronId
			#print w[layerId][neuronId][postNeuronId]
			#print d[layerId - 1][postNeuronId]
			#print d[layerId][postNeuronId]
			wdSum += w[layerId][neuronId][postNeuronId] * d[layerId][postNeuronId]
		if neuronId > 0:
			d[layerId - 1][neuronId - 1] = td[layerId - 1][neuronId - 1](h[layerId - 1][neuronId - 1]) * wdSum


def backwardPropLayerGradients(layerId, y, y_T):
	e_deriv = error_deriv(y, y_T)
	for neuronId in range(n[layerId] + 1):
		SWithBias = [biasS] + S[layerId]
		for postNeuronId in range(n[layerId + 1]):
			grad[layerId][neuronId][postNeuronId] += e_deriv * d[layerId][postNeuronId] * SWithBias[neuronId]
			# use these for online learning (vs batch-learning)
			#w[layerId][neuronId][postNeuronId] = w[layerId][neuronId][postNeuronId] - (learnRate * grad[layerId][neuronId][postNeuronId])
			#grad[layerId][neuronId][postNeuronId] = 0.0


def backwardPropStep(dataIndex, y, y_T):
	# local errors of the output neurons
	d[1][0] = td[1][0](h[1][0])

	# local errors for hidden layers
	llei = range(1, len(n) - 1)
	llei.reverse()
	for layerId in llei:
		backwardPropLayerLocalErrors(layerId)

	# gradients for all weights
	lgi = range(len(n) - 1)
	lgi.reverse()
	for layerId in lgi:
		backwardPropLayerGradients(layerId, y, y_T)


def visualize():
	xVals = deciRange(0.0, 1.0, 0.01)

	yValsToApprox = []
	for x in xVals:
		yValsToApprox.append(toApproximate(x))

	yValsMlp = []
	for x in xVals:
		yValsMlp.append(forwardPropStep(x))

	# Create the mathplot graph
	import pylab
	pylab.xlabel("x")
	pylab.ylabel("y / y_t")
	pylab.plot(xVals, yValsToApprox)
	pylab.scatter(xVals, yValsMlp)
	pylab.scatter(inputs, outputs)
	#pylab.xlim(0.0, 1.0)
	pylab.show()




for iterationId in range(10000):

	print  "Starting training iteration %i ..." % (iterationId)

	#print  "x\ty\ty_T\terror"
	for dataIndex in range(len(inputs)):
		x = inputs[dataIndex][0]
		y_T = forwardPropStep(x)

		y = outputs[dataIndex][0]
		e = error(y, y_T)
		#print x, "\t", y, "\t", y_T, "\t", e

		backwardPropStep(dataIndex, y, y_T)

	# adjust weights & reset gradients
	lgi = range(len(n) - 1)
	lgi.reverse()
	for layerId in lgi:
		for neuronId in range(n[layerId] + 1):
			for postNeuronId in range(n[layerId + 1]):
				wDelta = learnRate * grad[layerId][neuronId][postNeuronId] / len(inputs)
				w[layerId][neuronId][postNeuronId] = w[layerId][neuronId][postNeuronId] + wDelta
				grad[layerId][neuronId][postNeuronId] = 0.0


# Visualize the current approximation quality
print  "\nVisualize the result ..."
visualize()
exit(0)


