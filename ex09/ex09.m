#! /usr/bin/octave -qf


global DATA = [
-1 -1 -1
2 2 1
-3 -3 -1 
1 2 1
1 3 1
1 4 1
];


function _distance = distance(p1, p2)
	_distance = norm(p1-p2);
endfunction

# 9.1 k Nearest Neighbors
function _knn = knn(point, k)
	global DATA;
	distances = [];
	n = length(DATA);
	for i = 1:n # iterate over all points
		p2 = [DATA(i); DATA(i+n)]; # get current point
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

knn([-5;-5], 3)
