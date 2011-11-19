#
# TU Berlin - Machine Intelligence I WS11/12 - Exercise 4
# A Simple MLP
# authors: Rolf Schroeder & Robin Vobruba
#

import math
import random
import time
import pylab
import sys
import getopt

random.seed()

maxIterations = 3000
adaptiveLearning = False
onlineLearning = False
nHidden = 3
learnRate = 0.5
samplesFile = "data.txt"

# Prints how to use this script to the screen
def usage():
    print """Usage: %s
    Approximate sin() with an MLP.

    -a           : Use adaptive learning (default: off)
    -n <num>     : (Max) number of iterations (default: %i)
    -o           : Use online-learning, instead of batch-learning (default: off)
    -h <num>     : Number of hidden neurons (default: %i)
    -l <num>     : (Initial) learning rate (default: %f)
    -t <file>    : training data file (default: %s)

    """ % (sys.argv[0], maxIterations, nHidden, learnRate, samplesFile)
    sys.exit(-1)
try:
    opts, files = getopt.getopt(sys.argv[1:],"an:oh:l:t:")
except getopt.GetoptError:
    usage()

for opt, arg in opts:
    if opt == '-a':
	adaptiveLearning = True
    if opt == '-n':
	maxIterations = int(arg)
    if opt == '-o':
	onlineLearning = True
    if opt == '-h':
	nHidden = int(arg)
    if opt == '-l':
	learnRate = float(arg)
    if opt == '-t':
	samplesFile = arg

# Define the networks component vars

# interval (from -alpha to alpha)
alpha = 0.5

if (adaptiveLearning):
	minConverged = 0.00001
	adaptiveLearnRateFactorGreaterThanOne = 1.02
	adaptiveLearnRateFactorSmallerThanOne = 0.5

biasS = 1.0 # the activitation of the bias neuron

# number of neurons per layer (NOT counting the bias)
# n[layer] = no.
n = [1, nHidden, 1]

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
tdo = []

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

def transfer_hidden_deriv_optimized(transferVal):
	tanhX = transferVal
	return 1 - tanhX**2

def transfer_output(x):
	return x

def transfer_output_deriv(x):
	return 1

def transfer_output_deriv_optimized(transferVal):
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
for hiddenLayerId in range(1, len(n) - 1):
	t.append([transfer_hidden] * n[hiddenLayerId])
	td.append([transfer_hidden_deriv] * n[hiddenLayerId])
	tdo.append([transfer_hidden_deriv_optimized] * n[hiddenLayerId])
lastLayerId = len(n) - 1
t.append([transfer_output] * n[lastLayerId])
td.append([transfer_output_deriv] * n[lastLayerId])
tdo.append([transfer_output_deriv_optimized] * n[lastLayerId])

# activities
for layerId in range(len(n)):
	S.append([0.0] * n[layerId])


###### LEARNING ######

# Read the data file
try:
	data_f = open(samplesFile, 'r')
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
			# sum up the error of each following neuron multiplied with it's weight
			wdSum += w[layerId][neuronId][postNeuronId] * d[layerId][postNeuronId]
		if neuronId > 0:
			# the delta of this neuron (id -1 due to bias index)
			#d[layerId - 1][neuronId - 1] = td[layerId - 1][neuronId - 1](h[layerId - 1][neuronId - 1]) * wdSum
			# optimized version
			d[layerId - 1][neuronId - 1] = td[layerId - 1][neuronId - 1](S[layerId][neuronId - 1]) * wdSum


def backwardPropLayerGradients(layerId, y, y_T):
	SWithBias = [biasS] + S[layerId] # activition of this layer
	e_deriv = error_deriv(y, y_T)
	for neuronId in range(n[layerId] + 1): # for each neuron in this layer + the bias
		for postNeuronId in range(n[layerId + 1]): # for each neuron in the next layer
			grad[layerId][neuronId][postNeuronId] += e_deriv * d[layerId][postNeuronId] * SWithBias[neuronId]
			if onlineLearning:
				wDelta = learnRate * grad[layerId][neuronId][postNeuronId]
				w[layerId][neuronId][postNeuronId] = w[layerId][neuronId][postNeuronId] + wDelta
				grad[layerId][neuronId][postNeuronId] = 0.0


def backwardPropStep(dataIndex, y, y_T):
	# local errors of the output neurons
	lastLayerId = len(n) - 2
	d[lastLayerId][0] = td[lastLayerId][0](h[lastLayerId][0])

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
	pylab.subplot(3,1,1)
	xVals = deciRange(0.0, 1.0, 0.01)

	yValsToApprox = []
	for x in xVals:
		yValsToApprox.append(toApproximate(x))

	yValsMlp = []
	yValsSs = [[] for r in range(n[1])]
	for x in xVals:
		yValsMlp.append(forwardPropStep(x))
		for activity in range(len(S[1])):
			yValsSs[activity].append(S[1][activity])
	pylab.xlabel("x")
	pylab.ylabel("y")
	pylab.plot([0.0, 1.0], [0.0, 0.0], color='black')
	pylab.plot(xVals, yValsToApprox, color='red', label='sin(2*PI*x)')
	pylab.scatter(xVals, yValsMlp, color='blue', label='y_T(x)')
	pylab.scatter(inputs, outputs, color='yellow', label='samples')
	for activity in range(len(yValsSs)):
		plotLabel = 'S hidden %i' % (activity)
		pylab.plot(xVals, yValsSs[activity], color='green', label=plotLabel)
	pylab.xlim(-0.05, 1.5)
	pylab.legend()

	# plot ET over iterations
	pylab.subplot(3,1,2)
	xVals = range(len(ETs))
	yVals = ETs
	pylab.xlabel("iterations")
	pylab.ylabel("y")
	pylab.plot(xVals, yVals, color='blue', label="ET")
	pylab.legend()

	# plot deltaET over iterations
	pylab.subplot(3,1,3)
	pylab.xlabel("iterations")
	pylab.ylabel("y")
	xVals2 = range(len(deltaETs))
	yVals2 = deltaETs
	pylab.plot(xVals2, yVals2, color='green', label="delta-ET")
	pylab.ylim(0.0, 0.0003)
	pylab.legend()

	pylab.show()

ETs = []
deltaETs = []
ETlast = 1.0
for iterationId in range(maxIterations):

	print  "Starting training iteration %i ..." % (iterationId)

	#print  "x\ty\ty_T\terror"
	ETCur = 0.0
	N = len(inputs)
	sampleIndices = range(N)
	if onlineLearning:
		random.shuffle(sampleIndices)
	for dataIndex in sampleIndices: # for each datapoint
		x = inputs[dataIndex][0] # this datapoint's input
		y_T = forwardPropStep(x) # propagata info

		y = outputs[dataIndex][0] # the desired output
		e = error(y, y_T) # difference between desired and actual output
		ETCur += e
		#print x, "\t", y, "\t", y_T, "\t", e

		backwardPropStep(dataIndex, y, y_T) # calculate local errors (deltas)
	ETCur = ETCur / N
	ETs.append(ETCur)
	deltaET = ETCur - ETlast
	deltaETs.append(math.fabs(deltaET))

	if (adaptiveLearning):
		#print ETCur, ETlast, deltaET
		# stop if needed
		if (math.fabs(deltaET / ETCur)  < minConverged):
			print("converged")
			break
		elif (deltaET < 0):
			learnRate = learnRate * adaptiveLearnRateFactorGreaterThanOne
		else:
			learnRate = learnRate * adaptiveLearnRateFactorSmallerThanOne
	ETlast = ETCur

	# adjust weights & reset gradients
	if not onlineLearning:
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


