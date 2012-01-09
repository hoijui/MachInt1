#! /usr/bin/octave -qf

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

function _knn = knn(dataTrainingC, point, k)
	distances = [];
	n = length(dataTrainingC);
	for i = 1:n # iterate over all training points
		x = [dataTrainingC(i, 1), dataTrainingC(i, 2)]; # training data point
		t = dataTrainingC(i, 3); # label
		# add distance and target of current point
		distances = [distances; distance(point, x), t];
	endfor
	distances = sortrows(distances); # sort rows (by first column -> sort by distance ascending)
	distances(k+1:end,:)=[]; # delete everything except the first k rows
	s = sum(distances, 1); # sum along the cols
	s(:,1) = []; # keep only the label sum
	_knn = round(s / k);
endfunction


# 9.2 Parzen Windows

function _pn = pn(dataTrainingC, point, parzenSigma)
	classesWeighted = [0, 0];
	n = length(dataTrainingC);
	for i = 1:n # iterate over all training points
		x = [dataTrainingC(i, 1), dataTrainingC(i, 2)]; # training data point
		t = dataTrainingC(i, 3); # label
		# add distance and target of current point
		classesWeighted(t) += 1 / sqrt(2*pi*parzenSigma^2) * exp(-distance(point, x)^2 / (2*parzenSigma^2));
	endfor
	if classesWeighted(1) > classesWeighted(2)
		_pn = 1;
	else
		_pn = 2;
	endif
endfunction



# 9.3 RBF Network

function _phi = phi(x, t, sigma)
	_phi = exp(-1 * (x - t)^2 / 2 * sigma^2);
endfunction

# calculate the k centroids
function _centroids = kmeans(dataTrainingC, k, DATA)
	n = length(dataTrainingC);
	# pick random data points as initial centroids
	t = [];
	for i = 1:k
		index = unidrnd(n);
		t = [t, get_point(dataTrainingC, index)];
	endfor
	# update cenroids
	for i = 1:(10 * n)
		index = unidrnd(n);
		# choose data point
		x = get_point(dataTrainingC, index);

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


function plotClsDist(dataTrainingP, testC1, testC2, name)
	global mySamples;

	f = figure('Visible', 'off');
	hold on
	plot(testC1'(1,:), testC1'(2,:), 'g*');
	plot(testC2'(1,:)', testC2'(2,:), 'm*');

	plot(dataTrainingP(1,:), dataTrainingP(2,:), 'r*');
	plot(dataTrainingP(3,:), dataTrainingP(4,:), 'b*');
	title("Verteilung");
	legend(["C1"; "C2"]);
	print(strcat("out_", name, ".png"), "-dpng");
	hold off
endfunction


function _classifiedGrid = classifyGrid(dataTrainingC, classifier)
	global stepSize;

	_classifiedGrid = [];

	steps = -1 : stepSize : 2;
	for x = steps
		for y = steps
			tdi = length(_classifiedGrid) + 1;
			_classifiedGrid(tdi, 1) = x;
			_classifiedGrid(tdi, 2) = y;
			_classifiedGrid(tdi, 3) = feval(classifier, dataTrainingC, [x, y]);
		endfor
	endfor
endfunction



function _testClassData = separateTestDataIntoClasses(classifiedGrid, wantedClass)
	_testClassData = [];
	for tdi = 1:length(classifiedGrid)
		if classifiedGrid(tdi, 3) == wantedClass
			newInd = length(_testClassData) + 1;
			_testClassData(newInd, 1) = classifiedGrid(tdi, 1);
			_testClassData(newInd, 2) = classifiedGrid(tdi, 2);
			_testClassData(newInd, 3) = classifiedGrid(tdi, 3);
		end
	endfor
endfunction

function _cls = classifierKnn(dataTrainingC, point)
	global k;
	_cls = knn(dataTrainingC, point, k);
endfunction

function _cls = classifierParzen(dataTrainingC, point)
	global parzenSigma;
	_cls = pn(dataTrainingC, point, parzenSigma);
endfunction

function _cls = plotClassifier(dataTrainingC, dataTrainingP, classifier, nameSuffix)
	classifiedGrid = classifyGrid(dataTrainingC, classifier);
	testClassData1 = separateTestDataIntoClasses(classifiedGrid, 1);
	testClassData2 = separateTestDataIntoClasses(classifiedGrid, 2);

	name = strcat(classifier, "_", nameSuffix);
	plotClsDist(dataTrainingP, testClassData1, testClassData2, name);
endfunction


function plotKnn(dataTrainingC, dataTrainingP, myK)
	global k;
	k = myK;
	plotClassifier(dataTrainingC, dataTrainingP, 'classifierKnn', strcat("k_", num2str(k)));
endfunction

function plotParzen(dataTrainingC, dataTrainingP, mySigma)
	global parzenSigma;
	parzenSigma = mySigma;
	plotClassifier(dataTrainingC, dataTrainingP, 'classifierParzen', strcat("sigma_", num2str(parzenSigma)));
endfunction



dataTrainingP = generateSamples();

dataTrainingC = [
	dataTrainingP(1,:)' dataTrainingP(2,:)' 1 * ones(nSampPerClass, 1), zeros(nSampPerClass, 1);
	dataTrainingP(3,:)' dataTrainingP(4,:)' 2 * ones(nSampPerClass, 1), zeros(nSampPerClass, 1)];

global stepSize = 0.3;



plotKnn(dataTrainingC, dataTrainingP, 1);
plotKnn(dataTrainingC, dataTrainingP, 5);
plotKnn(dataTrainingC, dataTrainingP, 25);

plotParzen(dataTrainingC, dataTrainingP, 0.01);
plotParzen(dataTrainingC, dataTrainingP, 0.1);
plotParzen(dataTrainingC, dataTrainingP, 0.5);


