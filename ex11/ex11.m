#! /usr/bin/octave -qf

# add libsvm to octave's path
path(path,"./libsvm-3.11/matlab")

# libsvm options from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
% 
% The k in the -g option means the number of attributes in the input data.



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
	title(strrep(strcat("Distribution_", strrep(name, "classifier", "")), "_", " "));
	legend(["test-C_1"; "test-C_2"; "training-C_1"; "training-C_2"]);
	print(strcat("out_", strrep(name, ".", ""), ".png"), "-dpng");
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
			isInit = (x == -1) && (y == -1);
			_classifiedGrid(tdi, 3) = feval(classifier, dataTrainingC, [x, y], isInit);
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
function _cls = plotClassifier(dataTrainingC, dataTrainingP, classifier, nameSuffix)
	classifiedGrid = classifyGrid(dataTrainingC, classifier);
	testClassData1 = separateTestDataIntoClasses(classifiedGrid, 1);
	testClassData2 = separateTestDataIntoClasses(classifiedGrid, 2);

	name = strcat(classifier, "_", nameSuffix);
	plotClsDist(dataTrainingP, testClassData1, testClassData2, name);
endfunction



dataTrainingP = generateSamples();

dataTrainingC = [
	dataTrainingP(1,:)' dataTrainingP(2,:)' 1 * ones(nSampPerClass, 1);
	dataTrainingP(3,:)' dataTrainingP(4,:)' -1 * ones(nSampPerClass, 1)];
