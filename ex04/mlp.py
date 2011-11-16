#
# TU Berlin - Machine Intelligence I WS11/12 - Exercise 4
# A Simple MLP
# authors: Rolf Schroeder & Robin Vobruba
#

import math, random, time, pylab

random.seed()


# Define the networks component vars

# interval (from -alpha to alpha)
alpha = 0.5

learnRate = 0.5
adaptiveLearning = False
if (adaptiveLearning):
	minConverged = 0.00001
	adaptiveLearnRateFactorGreaterThanOne = 1.02
	adaptiveLearnRateFactorSmallerThanOne = 0.5
maxIterations = 9000

biasS = 1.0 # the activitation of the bias neuron

# number of neurons per layer (NOT counting the bias)
# n[layer] = no.
n = [1, 3, 1]

# weights
# w[layerId][neuronFrom][neuronTo]
# neuronFrom == 0 -> the bias weight
w = []

# local inputs
# h[layerId][neuronId]
h = []

# transfer functions
# the matrices contain the corresponding function for each neuron in
# each layer
t = []
td = []

# local errors
d = []

# gradients
grad = []

# activities
# S[layerId][neuronfrom]
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
	SWithBias = [biasS] + S[layerId - 1] # activitation from the prev. layer
	for neuronId in range(n[layerId]): # for each neuron in this layer
		nPre = n[layerId - 1]
		# bias * thresholdWeight
		hCur = 0.0 # input sum for this neuron
		for preNeuronId in range(n[layerId - 1] + 1): # for each neuron of the prev. layer
			Spre = SWithBias[preNeuronId]
			wcur = w[layerId - 1][preNeuronId][neuronId]
			hCur = hCur + (Spre * wcur) # add prod. of prev. neuron activitation and it's weight
		h[layerId - 1][neuronId] = hCur # this neuron's total input
		S[layerId][neuronId] = t[layerId - 1][neuronId](hCur) # this neuron's activitation



def forwardPropStep(x):
	# "activities" of layer 0 (e.g. inputs)
	S[0] = [x]

	# activities for hidden layers
	for layerId in range(len(n) - 1): # propagate info from layer to layer
		forwardPropLayer(layerId + 1)

	y_T = S[len(n) - 1][0] # the nn's final output

	return y_T


# Backwards propagation
def backwardPropLayerLocalErrors(layerId):
	for neuronId in range(n[layerId] + 1): # for each neuron in this layer
		wdSum = 0.0 # the sum of the weigthed delta from the following neurons
		for postNeuronId in range(n[layerId + 1]): # for each neuro in the next layer
			#print "layerId: ", layerId, "neuronId: ", neuronId, "postNeuronId: ", postNeuronId
			#print w[layerId][neuronId][postNeuronId]
			#print d[layerId - 1][postNeuronId]
			#print d[layerId][postNeuronId]

			# sum up the error of each following neuron multiplied with it's weight
			wdSum += w[layerId][neuronId][postNeuronId] * d[layerId][postNeuronId]
		if neuronId > 0:
			# the delta of this neuron (id -1 due to bias index)
			d[layerId - 1][neuronId - 1] = td[layerId - 1][neuronId - 1](h[layerId - 1][neuronId - 1]) * wdSum


def backwardPropLayerGradients(layerId, y, y_T):
	SWithBias = [biasS] + S[layerId] # activition of this layer
	e_deriv = error_deriv(y, y_T)
	for neuronId in range(n[layerId] + 1): # for each neuron in this layer + the bias
		for postNeuronId in range(n[layerId + 1]): # for each neuron in the next layer
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
	# plot hidden/output activitions, datapoints and current approx.
	pylab.subplot(2,1,1)
	xVals = deciRange(0.0, 1.0, 0.01)

	yValsToApprox = []
	for x in xVals:
		yValsToApprox.append(toApproximate(x))

	yValsMlp = []
	yValsSs = [[], [], []]
	for x in xVals:
		yValsMlp.append(forwardPropStep(x))
		for activity in range(len(S[1])):
			yValsSs[activity].append(S[1][activity])
	pylab.xlabel("x")
	pylab.ylabel("y / y_t")
	pylab.plot([0.0, 1.0], [0.0, 0.0], color='black')
	pylab.plot(xVals, yValsToApprox, color='red', label='sin()')
	pylab.scatter(xVals, yValsMlp, color='blue', label='S[2][0]')
	pylab.scatter(inputs, outputs, color='yellow', label='smpl')
	for activity in range(len(yValsSs)):
		plotLabel = 'S[1][%i]' % (activity)
		pylab.plot(xVals, yValsSs[activity], color='green', label=plotLabel)
	#pylab.xlim(0.0, 1.0)
	pylab.legend()

	# plot ET over iterations
	pylab.subplot(2,1,2)
	xVals = deciRange(1, len(ETs), 1)
	yVals = ETs
	pylab.xlabel("iterations")
	pylab.ylabel("ET")
	pylab.plot(xVals, yVals, label="ET")
	pylab.legend()
	pylab.show()

ETs = []
ETCur = 0.0
for iterationId in range(maxIterations):

	print  "Starting training iteration %i ..." % (iterationId)

	#print  "x\ty\ty_T\terror"
	ETlast = ETCur
	N = len(inputs)
	for dataIndex in range(N): # for each datapoint
		x = inputs[dataIndex][0] # this datapoint's input
		y_T = forwardPropStep(x) # propagata info

		y = outputs[dataIndex][0] # the desired output
		e = error(y, y_T) # difference between desired and actual output
		ETCur += e
		#print x, "\t", y, "\t", y_T, "\t", e

		backwardPropStep(dataIndex, y, y_T) # calculate local errors (deltas)
	ETCur = ETCur / N
	ETs.append(ETCur)


	if (adaptiveLearning):
		deltaET = ETCur - ETlast
		#print ETCur, ETlast, deltaET
		# stop if needed
		if ((deltaET / ETCur)  < minConverged):
			print("converged")
			break
		elif (deltaET < 0):
			learnRate = learnRate * adaptiveLearnRateFactorGreaterThanOne
		else:
			learnRate = learnRate * adaptiveLearnRateFactorSmallerThanOne

	# adjust weights & reset gradients
	lgi = range(len(n) - 1)
	lgi.reverse()
	for layerId in lgi: # iterate over layers (except input layer) in reversed order
		for neuronId in range(n[layerId] + 1): # for each neuron of this layer (+bias)
			for postNeuronId in range(n[layerId + 1]): # for each neuron in the next layer
				wDelta = learnRate * grad[layerId][neuronId][postNeuronId] / N
				w[layerId][neuronId][postNeuronId] = w[layerId][neuronId][postNeuronId] + wDelta
				grad[layerId][neuronId][postNeuronId] = 0.0


# Visualize the current approximation quality
print  "\nVisualize the result ..."
visualize()
exit(0)


