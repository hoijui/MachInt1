#! /usr/bin/octave -qf


#global DATA = [
#-1 -1 -1
#2 2 1
#-3 -3 -1
#1 2 1
#1 3 1
#1 4 1
#];
global DATA = [
1 1
1 3
2 2
3 1
3 3
#####
1 -1
1 -2
1.5 -1.5
2 -1
2 -2
######
-3 1
-3 2
-2.5 1.5
-2 1
-2 2
]


global mu = [0, 1; 1, 0; 0, 0; 1, 1];
global sigma = sqrt(0.1);
global nSampPerClass = 40;

# Generate samples
function _samples = generateSamples()
	global mu;
	global sigma;
	global nSampPerClass;
	_samples = [];
	samplesC1X = [];
	samplesC1Y = [];
	samplesC2X = [];
	samplesC2Y = [];

	subC1 = [[], []; [], []];
	subC2 = [[], []; [], []];
	for p = 1:2
		for d = 1:2
			myMu = mu(p, d);
			for s = 1:nSampPerClass
				subC1(p, d, s) = myMu + sigma*randn;
			endfor
		endfor
	endfor
	for p = 1:2
		for d = 1:2
			myMu = mu(2+p, d);
			for s = 1:nSampPerClass
				subC2(p, d, s) = myMu + sigma*randn;
			endfor
		endfor
	endfor

	for s = 1:nSampPerClass
		ci = 1 + mod(s, 2);
		x = subC1(ci, 1, s);
		y = subC1(ci, 2, s);
		samplesC1X(end+1) = x;
		samplesC1Y(end+1) = y;
	endfor

	for s = 1:nSampPerClass
		ci = 1 + mod(s, 2);
		x = subC2(ci, 1, s);
		y = subC2(ci, 2, s);
		samplesC2X(end+1) = x;
		samplesC2Y(end+1) = y;
	endfor

	_samples = [samplesC1Y; samplesC1X; samplesC2Y; samplesC2X];
endfunction

mySamples = generateSamples();

hold on
plot(mySamples(1,:), mySamples(2,:), 'r*');
plot(mySamples(3,:), mySamples(4,:), 'b*');
hold off
title("Verteilung")
legend(["C1"; "C2"]);
print("initial.png", "-dpng")


# measure distance between two data points
function _distance = distance(p1, p2)
	_distance = norm(p1-p2);
endfunction

# convenience function to access a datapoint
function _point = get_point(DATA, index)
	n = length(DATA);
	_point = [DATA(index); DATA(index + n)];
endfunction

# 9.1 k Nearest Neighbors
function _knn = knn(point, k)
	global DATA;
	distances = [];
	n = length(DATA);
	for i = 1:n # iterate over all points
		p2 = get_point(DATA, i); # get current point
		distances = [distances; distance(point, p2), DATA(i + 2 * n)]; # add distance and target of current point
	endfor
	distances = sortrows(distances); # sort rows (by first column -> sort by distance ascending)
	distances(k+1:end,:)=[]; # delete everything except the first k rows
	s = sum(distances, 1); # sum along the cols
	s(:,1) = []; # keep only the +1/-1 sum
	if (s < 0) # evalute the voting
		_knn = -1;
	else
		_knn = 1;
	endif
	
endfunction


# 9.2 Parzen Windows
function _pn = pn(point)


endfunction



# 9.3 RBF Network

function _phi = phi(x, t, sigma)
	_phi = exp(-1 * (x - t)^2 / 2 * sigma^2);
endfunction

# calculate the k centroids
function _centroids = kmeans(k, DATA)
	n = length(DATA);
	# pick random data points as initial centroids
	t = [];
	for i = 1:k
		index = unidrnd(n);
		t = [t, get_point(DATA, index)];
	endfor
	# update cenroids
	for i = 1:(10 * n)
		index = unidrnd(n);
		# choose data point
		x = get_point(DATA, index);

		# determine closest centroid
		distances = [];
		for j = 1:length(t)
			distances = [distances; distance(x, t(:,j)), j];
		endfor
		distances = sortrows(distances);
		jnearest = distances(1,2);
		nearest = t(:,jnearest);

		# update centroid
		nearest = nearest + (1 / n) * (x - nearest);
		t(:,jnearest) = nearest;
	endfor
	_centroids = t;
endfunction

kmeans(3, DATA);

