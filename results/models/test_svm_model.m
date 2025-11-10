% test_svm_model.m
clc; clear;

file = 'SVM_crack_detector_v1.mat';

data = load(file);

% Extract variables using the correct names
SVMModel = data.TrainedModel;
mu = data.mu_save;
sigma = data.sigma_save;

disp('✅ Model Loaded Successfully');

% Select test image
[img, path] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp'}, 'Select a test image');
I = imread(fullfile(path,img));

% Preprocess for feature extraction
if size(I,3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end

Igray = imresize(Igray, [200 200]);

% Extract features
fv = extractCrackFeatures(Igray);

% Standardize features (same used during training)
fv = (fv - mu) ./ sigma;

% Predict
[label, score] = predict(SVMModel, fv);

conf = max(score) * 100;

fprintf('\nPrediction: %s\nConfidence: %.2f%%\n', string(label), conf);
