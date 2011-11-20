#! /usr/bin/octave -qf


# X' = transposed(X)

global w = [0.5; 0.5]
global t = [-0.1, 0.5, 0.5]
global X = [1, 1, 1; (-1), 0.3, 2]

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
	X
	w
	t
	w' * X
	ret = 0.5 * sumsq(w' * X - t) +5;
endfunction

function ret = H()
	global X
	ret = 2 * X * X'
endfunction

# the gradient for this error
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
	alpha = - (gradient' * gradient ) / (gradient' * H() * gradient)
	w = w + alpha * gradient
endfunction


for i = 0:12
	#e = Error();
	g = gradient();
	#e
	#w
	g
	update_weights(0.2, g);
endfor
Error()
