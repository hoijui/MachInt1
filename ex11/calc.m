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

function _cls = classifierSvm(dataTrainingC, point, isInit)
	global model;
	global svmTrainOptions;
	if isInit
		d = 2; # number of input data dimensions
		trainingLabels = dataTrainingC(:, [3])'; # take only the 3rd row
		trainingInput = dataTrainingC(:, [1, 2])'; # take only the rows 1 and 2
		model = svmtrain(trainingLabels', trainingInput', svmTrainOptions);
		totalSV = model.totalSV
		save -append 'log' totalSV
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




function plotSvm(dataTrainingC, dataTrainingP, mySvmTrainOptions, name)
	global svmTrainOptions;
	svmTrainOptions = mySvmTrainOptions;
	save -append 'log' name
	plotClassifier(dataTrainingC, dataTrainingP, 'classifierSvm', name);
endfunction



dataTrainingP = generateSamples();

dataTrainingC = [
	dataTrainingP(1,:)' dataTrainingP(2,:)' 1 * ones(nSampPerClass, 1), zeros(nSampPerClass, 1);
	dataTrainingP(3,:)' dataTrainingP(4,:)' 2 * ones(nSampPerClass, 1), zeros(nSampPerClass, 1)];

global stepSize = 0.03;

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


exec112 = true;
exec113 = true;
exec114 = true;
exec115 = true;
unlink('log')

# 11.2 C-SVM with standard parameters
if exec112
	ex = "\n#########################\n# EX 11.2\n#########################\n"
	save -append 'log' ex
	plotSvm(dataTrainingC, dataTrainingP, "-q", "cSvmDefault");
endif


# 11.3 Parameter optimization
if exec113
	ex = "\n#########################\n# EX 11.3\n#########################\n"
	save -append 'log' ex
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
			save -append 'log' myC
			save -append 'log' myGamma
			save -append 'log' crossValidatedAccuracy
			gi = gi + 1;
		endfor
		ci = ci + 1;
	endfor
	surf(gammaExps, cExps, accuracies);
	title('SVM parameter optimization (RBF kernel)');
	legend(["accuracy"]);
	xlabel("gamma");
	ylabel("C");
	print('out_cSvmRbfParameterOptimization.png');
	save -append 'log' optimalParams
	save -append 'log' optimalAccuracy
endif



# 11.4 C-SVM with optimal parameters
# HACK There seems ot be a bug in the strcat function,
#   which makes it trim all sub-strings.
#   Thus we have to use the following trick.
if exec114
	ex = "\n#########################\n# EX 11.4\n#########################\n"
	save -append 'log' ex
	svmTrainOptions = strcat("-q -s 0 -c_", num2str(optimalParams(1)), " -g_", num2str(optimalParams(2)));
	svmTrainOptions = strrep(svmTrainOptions, "_", " ");
	plotSvm(dataTrainingC, dataTrainingP, svmTrainOptions, "cSvmRbfOptimalParams");
endif



# 11.5 C-SVM with polynomial kernels
if exec115
	ex = "\n#########################\n# EX 11.5\n#########################\n"
	save -append 'log' ex

	# 11.5 (2)
	plotSvm(dataTrainingC, dataTrainingP, "-q -t 1 -g 1 -r 1", "cSvmPolyDefault");

	# 11.5 (3)
	params = [];
	accuracies = [];
	cExps = -11 : 2 : 15;
	degreeExps = -5 : 2 : 7;
	ci = 1;
	optimalAccuracy = 0;
	optimalParams = [0, 0];
	for myCExp = cExps;
		gi = 1;
		for myDegreeExp = degreeExps;
			myC = 2^myCExp;
			myDegree = 2^myDegreeExp;
			global svmTrainOptions;
			# HACK There seems ot be a bug in the strcat function,
			#   which makes it trim all sub-strings.
			#   Thus we have to use the following trick.
			svmTrainOptions = strcat("-q -t 1 -g 1 -r 1 -v 16 -s 0 -c_", num2str(myC), " -d_", num2str(myDegree));
			svmTrainOptions = strrep(svmTrainOptions, "_", " ");
			# begin: init model
				trainingLabels = dataTrainingC(:, [3])'; # take only the 3rd row
				trainingInput = dataTrainingC(:, [1, 2])'; # take only the rows 1 and 2
				crossValidatedAccuracy = svmtrain(trainingLabels', trainingInput', svmTrainOptions);
			# end: init model
			params(end+1, [1, 2]) = [myC, myDegree];
			accuracies(ci, gi) = [crossValidatedAccuracy];
			if crossValidatedAccuracy > optimalAccuracy
				optimalParams = [myC, myDegree];
				optimalAccuracy = crossValidatedAccuracy;
			endif
			gi = gi + 1;
			save -append 'log' myC
			save -append 'log' myDegree
			save -append 'log' crossValidatedAccuracy
		endfor
		ci = ci + 1;
	endfor
	surf(degreeExps, cExps, accuracies);
	title('SVM parameter optimization (Polynomial kernel)');
	legend(["accuracy"]);
	xlabel("degree");
	ylabel("C");
	print('out_cSvmPolyParameterOptimization.png');
	save -append 'log' optimalParams
	save -append 'log' optimalAccuracy

	# 11.5 (4)
	# HACK There seems ot be a bug in the strcat function,
	#   which makes it trim all sub-strings.
	#   Thus we have to use the following trick.
	svmTrainOptions = strcat("-q -t 1 -g 1 -r 1 -s 0 -c_", num2str(optimalParams(1)), " -d_", num2str(optimalParams(2)));
	svmTrainOptions = strrep(svmTrainOptions, "_", " ");
	plotSvm(dataTrainingC, dataTrainingP, svmTrainOptions, "cSvmPolyOptimalParams");
endif


