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

