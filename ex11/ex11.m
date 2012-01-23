#! /usr/bin/octave -qf

# add libsvm to octave's path
path(path,"./libsvm-3.11/matlab")

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

dataTrainingP = generateSamples();

dataTrainingC = [
	dataTrainingP(1,:)' dataTrainingP(2,:)' 1 * ones(nSampPerClass, 1);
	dataTrainingP(3,:)' dataTrainingP(4,:)' -1 * ones(nSampPerClass, 1)];
