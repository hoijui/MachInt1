#! /usr/bin/octave -qf

#
# Here we are comparing 3 different learning algorithms,
# demonstrated on a simple perceptron.
# i)   gradient decsent
# ii)  line-search
# iii) conjugate gradient descent
#


# X' = transposed(X)


# Weight vector: w[0=>threshold, 1=>input-weight]
global wInit = 0.5;
global w = [0.0; 0.0]

# Output vector: t[trainingSampleId]
global t = [-0.1, 0.5, 0.5]

# Input matrix: X[trainingSampleId][0=>bias, 1=>input]
global input = [-1.0, 0.3, 2.0]
global X = [1.0, 1.0, 1.0; input]


# Returns the neuron's output/activity
function _y = y(x, w)
	yVec = w' * x
	_y = yVec(1) + yVec(2)
endfunction

#function _error = error(y, t)
#	_error = 0.5 * (y - t)^2
#endfunction

# the training error
function _Error = Error()
	global X
	global w
	global t
	_Error = 0.5 * sumsq(w' * X - t);
endfunction

# H = 2XX'
function _H = H()
	global X
	_H = 2 * X * X';
endfunction

# Returns the gradient for the current error
function _gradient = gradient()
	global w
	global X
	global t
	b = -2 * X * t';
	_gradient = H() * w + b;
endfunction

#################################

global w0
global w1
global e

function learn_init()
	global wInit;
	global w;
	global w0;
	global w1;
	global e;

	w(1) = unifrnd(-wInit, wInit);
	w(2) = unifrnd(-wInit, wInit);

	# save w0/1 for plotting
	w0 = [w(1)];
	w1 = [w(2)];

	e  = [Error()];
endfunction

# 1a) Gradient Descent
function learn_gradientDescent()
	global w;
	global w0;
	global w1;
	global e;

	iterations = 10;
	rate = 0.1;
	for i = 1:iterations
		g = gradient();
		w = w - rate * g;
		w0(end+1) = w(1);
		w1(end+1) = w(2);
		e(end+1)  = Error();
	endfor
endfunction

# 1b) Line search
function learn_lineSearch()
	global w;
	global w0;
	global w1;
	global e;

	iterations = 15;
	for i = 1:iterations
		g = gradient();
		alpha = - (g' * g) / (g' * H() * g);
		w = w + alpha * g;
		w0(end+1) = w(1);
		w1(end+1) = w(2);
		e(end+1)  = Error();
	endfor
endfunction

# 1c) Conjugate Gradient
function learn_conjugateGradient()
	global w;
	global w0;
	global w1;
	global e;

	gold = gradient();
	w = -gold;
	d = -gold;
	iterations = 2;
	for i = 1:iterations
		alpha = -(d' * gold) / (d' * H() * d);
		w = w + alpha * d;
		gnew = gradient();
		if (1 == 0)
			break;
		endif
		beta = -(gnew' * gnew) / (gold' * gold);
		d = gnew + beta * d;
		gold = gnew;

		w0(end+1) = w(1);
		w1(end+1) = w(2);
		e(end+1)  = Error();
	endfor
endfunction


function plotLearningResults(methodName)
	global input;
	global X;
	global t;
	global w;
	global w0;
	global w1;
	global e;

	# Plot the samples vs the approximation
	output = t
	approxOutput = [y(X(1), w), y(X(2), w), y(X(3), w)]
	title(strcat(methodName, " - target space - samples and approximation"))
	plot(input, [output; approxOutput])
	legend(["samples"; "approximation"]);
	print(strcat(methodName, "_approximation.png"), "-dpng")

	# Plot the weights evolution over the iteration steps
	plot(w0, w1)
	title(strcat(methodName, " - weight space - evolution"))
	xlabel("w0");
	ylabel("w1");
	w0Min = min(w0)
	w0Max = max(w0)
	w0Scale = w0Max - w0Min
	w1Min = min(w1)
	w1Max = max(w1)
	w1Scale = w1Max - w1Min
	wScaleMax = max([w0Scale, w1Scale])
	w0Max = w0Min + wScaleMax
	w1Max = w1Min + wScaleMax
	axis([w0Min w0Max w1Min w1Max])
	print(strcat(methodName, "_weightsEvolution.png"), "-dpng")

	# Plot the error evolution over the iteration steps
	plot(e)
	title(strcat(methodName, " - error - evolution"))
	print(strcat(methodName, "_errorsEvolution.png"), "-dpng")
endfunction


learn_init()
learn_gradientDescent()
plotLearningResults("gradientDescent")

learn_init()
learn_lineSearch()
plotLearningResults("lineSearch")

learn_init()
learn_conjugateGradient()
plotLearningResults("conjugateGradient")
