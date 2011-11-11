

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
# input and bias(x[0])
x = [1]
# interval (from -alpha to alpha)
alpha = 0.5

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
			

