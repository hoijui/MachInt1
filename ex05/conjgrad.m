#! /usr/bin/octave -qf


# X' = transposed(X)

exoa = 0;
exob = 0;
exoc = 0;

# Weight vector: w[0=>threshold, 1=>input-weight]
a = 0.5;
global w = [unifrnd(-a, a); unifrnd(-a, a)]

# Output vector: t[trainingSampleId]
global t = [-0.1, 0.5, 0.5]

# Input matrix: X[trainingSampleId][0=>bias, 1=>input]
global X = [1.0, 1.0, 1.0; (-1.0), 0.3, 2.0]


# the neuron's output
function ret = y(x, w)
	ret = w' * x
endfunction

#function ret = error(y, t)
#	ret = 0.5 * (y - t)^2
#endfunction

# the training error
function ret = Error()
	global X
	global w
	global t
	ret = 0.5 * sumsq(w' * X - t);
endfunction

# H = 2XX'
function ret = H()
	global X
	ret = 2 * X * X';
endfunction

# Returns the gradient for the current error
function ret = gradient()
	global w
	global X
	global t
	b = -2 * X * t';
	ret = H() * w + b;
endfunction

function update_weights_grad_descent(rate, gradient)
	global w;
	w = w - rate * gradient;
endfunction

function update_weights_line_search(gradient)
	global w
	alpha = - (gradient' * gradient ) / (gradient' * H() * gradient);
	w = w + alpha * gradient;
endfunction

#################################
# save w0/1 for plotting
w0 = [w(1)];
w1 = [w(2)];

# 1a) Gradient Descent
if (exoa == 1)
	iterations = 100;
	rate = 0.5;
	for i = 1:iterations
		g = gradient();
		update_weights_grad_descent(rate, g)
		w0(end+1) = w(1);
		w1(end+1) = w(2);
	endfor
endif

# 1b) Line search
if (exob == 1)
	iterations = 15;
	for i = 1:iterations
		g = gradient();
		update_weights_line_search(g)
		w0(end+1) = w(1);
		w1(end+1) = w(2);
	endfor
endif

# 1c) Conjugate Gradient
if (exoc == 1)
	gold = gradient();
	w = -gold;
	d = -gold;
	iterations = 20;
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

		#Error()
		w0(end+1) = w(1);
		w1(end+1) = w(2);
	endfor
endif

plot(w0, w1)
title("weight space")
xlabel("w0");
ylabel("w1");
axis([0 10 0 10])
#legend("uiae");
print("plot.png", "-dpng")
