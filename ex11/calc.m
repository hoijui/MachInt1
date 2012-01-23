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
function _point = getPoint(data, index)
	_point = [data(index, 1), data(index, 2)];
endfunction



# 9.1 k Nearest Neighbors

function _knn = knn(dataTrainingC, point, k)
	distances = [];
	n = length(dataTrainingC);
	for i = 1:n # iterate over all training points
		x = getPoint(dataTrainingC, i); # training data point
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
		x = getPoint(dataTrainingC, i); # training data point
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

function _phi = phi(x, rbfMu, rbfSigma)
	_phi = exp(-1 * distance(x, rbfMu)^2 / 2 * rbfSigma^2);
endfunction

# calculate the k centroids
function _centroids = kmeans(dataTrainingC, k)
	n = length(dataTrainingC);
	# pick random data points as initial centroids
	t = [];
	for i = 1:k
		index = unidrnd(n);
		t = [t; getPoint(dataTrainingC, index)];
	endfor
	# update cenroids
	for i = 1:(10 * n)
		index = unidrnd(n);
		# choose data point
		x = getPoint(dataTrainingC, index);

		# determine closest centroid
		distances = [];
		for j = 1:length(t)
			distances = [distances; distance(x, t(j,:)), j];
		endfor
		distances = sortrows(distances);
		jnearest = distances(1,2);
		nearest = t(jnearest,:);

		# update centroid
		nearest = nearest + (1 / n) * (x - nearest);
		t(jnearest,:) = nearest;
	endfor
	_centroids = t;
endfunction


function plotClsDist(dataTrainingP, testC1, testC2, name)
	global mySamples;
	global stepSize;

	distSymbSize = 40 * sqrt(stepSize);

	f = figure('Visible', 'off');
	hold on
	if length(testC1) > 0
		plot(testC1'(1,:), testC1'(2,:), '*', 'color', [1.0 0.5 0.5], 'markersize', distSymbSize);
	endif
	if length(testC2) > 0
		plot(testC2'(1,:)', testC2'(2,:), '*', 'color', [0.5 0.5 1.0], 'markersize', distSymbSize);
	endif

	plot(dataTrainingP(1,:), dataTrainingP(2,:), '*', 'color', [1.0 0.0 0.0]);
	plot(dataTrainingP(3,:), dataTrainingP(4,:), '*', 'color', [0.0 0.0 1.0]);
	title(strrep(strcat("Distribution_", strrep(strrep(name, "multiClassifier", ""), "classifier", "")), "_", " "));
	legend(["test-C_1"; "test-C_2"; "training-C_1"; "training-C_2"]);
	print(strcat("out_", strrep(name, ".", ""), ".png"), "-dpng");
	hold off
endfunction


function _classifierGrid = createClassifierGrid(dataTrainingC)
	global stepSize;

	"creating classifier grid ..."
	_classifierGrid = [];
	steps = -1 : stepSize : 2;
	for x = steps
		for y = steps
			_classifierGrid(end+1, [1, 2]) = [x, y];
		endfor
	endfor
	"creating classifier grid done."
endfunction

function _classifierGridLabels = createClassifierGridLabels(classifierGrid)

	"creating classifier grid labels ..."
	_classifierGridLabels = [];
	for cgi = 1 : length(classifierGrid')
		x = sign(classifierGrid(cgi, 1) - 0.5);
		if x == 0; x = -1; endif # cause sign(0.0) == 0 :/
		y = sign(classifierGrid(cgi, 2) - 0.5);
		if y == 0; y = -1; endif # cause sign(0.0) == 0 :/
		x = (-x + 1) / 2;
		y = ( y + 1) / 2;
		c = xor(x, y) + 1;
		_classifierGridLabels(end+1) = c;
	endfor
	"creating classifier grid labels done."
endfunction



function _classifiedGrid = classifyGrid(dataTrainingC, classifier)
	global classifierGrid;

	_classifiedGrid = [];

	for xNy = classifierGrid'
		x = xNy(1);
		y = xNy(2);
		tdi = length(_classifiedGrid) + 1;
		_classifiedGrid(tdi, 1) = x;
		_classifiedGrid(tdi, 2) = y;
		isInit = (x == -1) && (y == -1);
		_classifiedGrid(tdi, 3) = feval(classifier, dataTrainingC, [x, y], isInit);
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

function _cls = classifierKnn(dataTrainingC, point, isInit)
	global k;
	_cls = knn(dataTrainingC, point, k);
endfunction

function _cls = classifierParzen(dataTrainingC, point, isInit)
	global parzenSigma;
	_cls = pn(dataTrainingC, point, parzenSigma);
endfunction

function _cls = classifierRbf(dataTrainingC, point, isInit)
	global rbfK;
	global rbfSigma;
	global rbfMus;
	global rbfX;
	global rbfW;
	global rbfWC1;
	global rbfWC2;
	if isInit
		rbfMus = kmeans(dataTrainingC, rbfK);
		rbfX = dataTrainingC;
		rbfX(:,3:end)=[]; # delete the labels
		phiMatrix = [];
		for j = 1:rbfK
			phiCol = [];
			for alpha = 1:length(rbfX)
					phiCol = [phiCol; phi(rbfX(alpha), rbfMus(j), rbfSigma)];
			endfor
			phiMatrix = [phiMatrix phiCol];
		endfor
		% once we are here, phiMatrix is 80x4

		phiMatrix = [phiMatrix ones(length(rbfX), 1)]; % add bias, phiMatrix is now 80x5
		% get labels
		rbfT = dataTrainingC;
		rbfT(:,4)=[]; # delete the filler
		rbfT(:,1:2)=[]; # delete the data
		% comute weight vector
		%rbfW = pinv(phiMatrix) * rbfT;
		half = length(rbfX) / 2; % this is hard coded, won't work for more than 2 classes
		rbfTC1 = [ones(half, 1); zeros(half, 1)]; % hard coded: we suppose, that the first half data points are class1, the second half class two
		rbfTC2 = [zeros(half, 1); ones(half, 1)];
		rbfWC1 = pinv(phiMatrix) * rbfTC1;
		rbfWC2 = pinv(phiMatrix) * rbfTC2;
	endif

	% compute output
	%_y = 0;
	_yC1 = 0;
	_yC2 = 0;
	for j = 1:(rbfK)
		phiJ = phi(point, rbfMus(j), rbfSigma);
		%_y += rbfW(j) * phiJ;
		_yC1 += rbfWC1(j) * phiJ;
		_yC2 += rbfWC2(j) * phiJ;
	endfor
	% add bias
	%_y += _y + rbfW(rbfK+1) * 1.0;
	_yC1 += _yC1 + rbfWC1(rbfK+1) * 1.0;
	_yC2 += _yC2 + rbfWC2(rbfK+1) * 1.0;

	%if _y < 1.5
	if _yC1 < _yC2
		_cls = 1;
	else
		_cls = 2;
	endif
endfunction

function _cls = classifierSvm(dataTrainingC, point, isInit)
	global model;
	global svmTrainOptions;
	if isInit
		d = 2; # number of input data dimensions
		trainingLabels = dataTrainingC(:, [3])'; # take only the 3rd row
		trainingInput = dataTrainingC(:, [1, 2])'; # take only the rows 1 and 2
		model = svmtrain(trainingLabels', trainingInput', svmTrainOptions);
	endif
	res = svmpredict(123.456, point, model);
	_cls = res(1);
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

function plotParzen(dataTrainingC, dataTrainingP, mySigma2)
	global parzenSigma;
	parzenSigma = sqrt(mySigma2);
	plotClassifier(dataTrainingC, dataTrainingP, 'classifierParzen', strcat("sigma2_", num2str(mySigma2)));
endfunction

function plotRbf(dataTrainingC, dataTrainingP, myK, mySigma)
	global rbfK;
	global rbfSigma;
	rbfK = myK;
	rbfSigma = mySigma;
	plotClassifier(dataTrainingC, dataTrainingP, 'classifierRbf', strcat("k_", num2str(rbfK), "_sigma_", num2str(rbfSigma)));
endfunction

function plotSvm(dataTrainingC, dataTrainingP, mySvmTrainOptions)
	global svmTrainOptions;
	svmTrainOptions = mySvmTrainOptions;
	plotClassifier(dataTrainingC, dataTrainingP, 'classifierSvm', svmTrainOptions);
endfunction



dataTrainingP = generateSamples();

dataTrainingC = [
	dataTrainingP(1,:)' dataTrainingP(2,:)' 1 * ones(nSampPerClass, 1), zeros(nSampPerClass, 1);
	dataTrainingP(3,:)' dataTrainingP(4,:)' 2 * ones(nSampPerClass, 1), zeros(nSampPerClass, 1)];

global stepSize = 0.1;

global classifierGrid = createClassifierGrid(dataTrainingC);
global classifierGridLabels = createClassifierGridLabels(classifierGrid);




for myK = [1, 5, 25]
	#plotKnn(dataTrainingC, dataTrainingP, myK);
endfor

for mySigma2 = [0.01, 0.1, 0.5]
	#plotParzen(dataTrainingC, dataTrainingP, mySigma2);
endfor

for myK = [4, 8]
	for mySigma = [0.01, 0.02]
		#plotRbf(dataTrainingC, dataTrainingP, myK, mySigma);
	endfor
endfor


# 11.2 C-SVM with standard parameters
plotSvm(dataTrainingC, dataTrainingP, "-q");


# 11.3 Parameter optimization
params = [];
accuracies = [];
cExps = -7 : 2 : 15;
gammaExps = -11 : 2 : 7;
ci = 1;
optimalAccuracy = 0;
optimalParams = [0, 0];
for myCExp = cExps;
	gi = 1;
	for myGammaExp = gammaExps;
		myC = 2^myCExp;
		myGamma = 2^myGammaExp;
		global svmTrainOptions;
		# HACK There seems ot be a bug in the strcat function,
		#   which makes it trim all sub-strings.
		#   Thus we have to use the following trick.
		svmTrainOptions = strcat("-q -v 16 -s 0 -c_", num2str(myC), " -g_", num2str(myGamma));
		svmTrainOptions = strrep(svmTrainOptions, "_", " ");
		# begin: init model
			trainingLabels = dataTrainingC(:, [3])'; # take only the 3rd row
			trainingInput = dataTrainingC(:, [1, 2])'; # take only the rows 1 and 2
			crossValidatedAccuracy = svmtrain(trainingLabels', trainingInput', svmTrainOptions);
		# end: init model
		params(end+1, [1, 2]) = [myC, myGamma];
		accuracies(ci, gi) = [crossValidatedAccuracy];
		if crossValidatedAccuracy > optimalAccuracy
			optimalParams = [myC, myGamma];
			optimalAccuracy = crossValidatedAccuracy;
		endif
		gi = gi + 1;
	endfor
	ci = ci + 1;
endfor
surf(gammaExps, cExps, accuracies);
title('SVM parameter optimization (RBF kernel)');
legend(["accuracy"]);
xlabel("gamma");
ylabel("C");
print('out_parameterOptimization.png');



# 11.4 C-SVM with optimal parameters
# HACK There seems ot be a bug in the strcat function,
#   which makes it trim all sub-strings.
#   Thus we have to use the following trick.
svmTrainOptions = strcat("-q -s 0 -c_", num2str(optimalParams(1)), " -g_", num2str(optimalParams(2)));
svmTrainOptions = strrep(svmTrainOptions, "_", " ");
plotSvm(dataTrainingC, dataTrainingP, svmTrainOptions);

