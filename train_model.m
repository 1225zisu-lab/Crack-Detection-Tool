function train_model()
% TRAIN_MODEL  Train optimized SVM (deep optimization) for crack detection.
% Expects:
%  - extractFeatures(I) that returns 1x64 numeric features
%  - dataset folders:
%       data/crack_dataset/crack/    -> files like crack_1.jpg
%       data/crack_dataset/no_crack/ -> files like no_crack_1.jpg
%
% MATLAB R2024b recommended (uses OptimizeHyperparameters)

clc; clear; close all;
rng(1);

%% ---------- User settings ----------
dataRoot = fullfile(pwd,'data','crack_dataset');
crackDir = fullfile(dataRoot,'crack');
intactDir = fullfile(dataRoot,'no_crack');

% Filename patterns (confirmed by you)
crackPattern = 'crack_*.jpg';
intactPattern = 'no_crack_*.jpg';

% Feature length (must match extractFeatures)
FV_LEN = 64;

% HoldOut ratio (70% train / 30% test)
holdOutRatio = 0.30;

% Deep optimization iterations
maxEvals = 40;

% Output model filename
outDir = fullfile(pwd,'results','models');
if ~exist(outDir,'dir'), mkdir(outDir); end
saveFile = fullfile(outDir, 'SVM_crack_detector_v1.mat');

%% ---------- Collect files ----------
crackFiles = dir(fullfile(crackDir, crackPattern));
intactFiles = dir(fullfile(intactDir, intactPattern));

nC = numel(crackFiles);
nI = numel(intactFiles);
total = nC + nI;

if total == 0
    error('No images found. Check dataRoot and filename patterns.');
end
fprintf('Found %d crack images and %d no_crack images (total %d)\n', nC, nI, total);

%% ---------- Extract features (per-image) ----------
X = zeros(total, FV_LEN);
Y = strings(total,1);
idx = 1;

% helper to safely extract and ensure length
pad_or_trim = @(v) ( (numel(v) < FV_LEN) * [v(:)' zeros(1,FV_LEN-numel(v))] + ...
                     (numel(v) >= FV_LEN) * v(1:FV_LEN) );

% cracked images
for k=1:nC
    fname = crackFiles(k).name;
    fpath = fullfile(crackFiles(k).folder, fname);
    try
        I = imread(fpath);
        fv = extractFeatures(I);
        if ~isnumeric(fv) || isempty(fv)
            warning('extractFeatures returned invalid for %s', fname);
            fv = nan(1,FV_LEN);
        end
    catch ME
        warning('Failed to read/extract %s : %s', fname, ME.message);
        fv = nan(1,FV_LEN);
    end
    fv = pad_or_trim(fv);
    X(idx,:) = fv;
    Y(idx) = "crack";
    idx = idx + 1;
end

% no_crack images
for k=1:nI
    fname = intactFiles(k).name;
    fpath = fullfile(intactFiles(k).folder, fname);
    try
        I = imread(fpath);
        fv = extractFeatures(I);
        if ~isnumeric(fv) || isempty(fv)
            warning('extractFeatures returned invalid for %s', fname);
            fv = nan(1,FV_LEN);
        end
    catch ME
        warning('Failed to read/extract %s : %s', fname, ME.message);
        fv = nan(1,FV_LEN);
    end
    fv = pad_or_trim(fv);
    X(idx,:) = fv;
    Y(idx) = "no_crack";
    idx = idx + 1;
end

% Trim any unused prealloc (shouldn't be needed but safe)
if idx <= size(X,1)
    X(idx:end,:) = [];
    Y(idx:end) = [];
end

% Convert labels to categorical and ensure two distinct classes
Y = categorical(Y);
uniqueLabels = categories(Y);
if numel(uniqueLabels) < 2
    error('Only one label detected. Found: %s. Ensure you have both classes present.', strjoin(uniqueLabels,','));
end

%% ---------- Clean NaN/Inf rows ----------
badRows = any(isnan(X),2) | any(isinf(X),2);
if any(badRows)
    fprintf('Removing %d rows with NaN/Inf features.\n', sum(badRows));
    X(badRows,:) = [];
    Y(badRows) = [];
end

if isempty(X)
    error('No valid feature rows remain after cleaning. Fix extractFeatures or dataset.');
end

%% ---------- Shuffle ----------
perm = randperm(size(X,1));
X = X(perm,:);
Y = Y(perm);

%% ---------- Split into train/test (HoldOut with fallback) ----------
try
    cv = cvpartition(Y,'HoldOut',holdOutRatio);
    trainIdx = training(cv);
    testIdx  = test(cv);
    % ensure both classes exist in training
    if numel(unique(Y(trainIdx))) < 2 || sum(trainIdx) < 4
        error('HoldOut produced insufficient or single-class training set.');
    end
    Xtrain = X(trainIdx,:);
    Ytrain = Y(trainIdx);
    Xtest  = X(testIdx,:);
    Ytest  = Y(testIdx);
    fprintf('Using HoldOut: %d train, %d test\n', sum(trainIdx), sum(testIdx));
catch
    fprintf('HoldOut unsuitable -> using 5-Fold CV first fold as fallback.\n');
    cvk = cvpartition(Y,'KFold',5);
    trainIdx = training(cvk,1);
    testIdx  = test(cvk,1);
    Xtrain = X(trainIdx,:);
    Ytrain = Y(trainIdx);
    Xtest  = X(testIdx,:);
    Ytest  = Y(testIdx);
    fprintf('Using 5-fold(1): %d train, %d test\n', sum(trainIdx), sum(testIdx));
end

disp('Label counts (train):'); disp(tabulate(Ytrain));
disp('Label counts (test):');  disp(tabulate(Ytest));

%% ---------- Standardize (mean/std) ----------
mu = mean(Xtrain,1);
sigma = std(Xtrain,0,1);
zeroSigma = sigma == 0;
if any(zeroSigma)
    fprintf('Warning: %d features have zero std. Setting sigma=1 for those features.\n', sum(zeroSigma));
    sigma(zeroSigma) = 1;
end

XtrainS = (Xtrain - mu) ./ sigma;
XtestS  = (Xtest  - mu) ./ sigma;

% sanity check
if any(any(isnan(XtrainS))) || any(any(isnan(XtestS)))
    error('NaNs present after standardization. Inspect feature generation.');
end

%% ---------- Hyperparameter optimization for multiple kernels ----------
kernelList = {'rbf','linear','polynomial'};
bestModel = [];
bestLoss = inf;
bestKernel = '';

% Optimization options common
optOpts = struct(...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'MaxObjectiveEvaluations', maxEvals, ...
    'ShowPlots', false, ...
    'Verbose', 0);

% For reproducibility
rng(2);

for ki = 1:numel(kernelList)
    kf = kernelList{ki};
    fprintf('\n=== Tuning kernel: %s ===\n', upper(kf));
    try
        switch kf
            case 'rbf'
                % Optimize BoxConstraint & KernelScale
                vars = {'BoxConstraint','KernelScale'};
                M = fitcsvm(XtrainS, Ytrain, ...
                    'KernelFunction','rbf', ...
                    'Standardize', false, ...
                    'OptimizeHyperparameters', vars, ...
                    'HyperparameterOptimizationOptions', optOpts);
            case 'linear'
                % Optimize BoxConstraint only
                vars = {'BoxConstraint'};
                M = fitcsvm(XtrainS, Ytrain, ...
                    'KernelFunction','linear', ...
                    'Standardize', false, ...
                    'OptimizeHyperparameters', vars, ...
                    'HyperparameterOptimizationOptions', optOpts);
            case 'polynomial'
                % Optimize BoxConstraint, KernelScale, PolynomialOrder
                % (PolynomialOrder optimization supported in R2024b)
                vars = {'BoxConstraint','KernelScale','PolynomialOrder'};
                M = fitcsvm(XtrainS, Ytrain, ...
                    'KernelFunction','polynomial', ...
                    'Standardize', false, ...
                    'OptimizeHyperparameters', vars, ...
                    'HyperparameterOptimizationOptions', optOpts);
            otherwise
                error('Unknown kernel %s', kf);
        end

        % Get cross-validated loss if possible
        try
            CVSVM = crossval(M,'KFold',5);
            loss = kfoldLoss(CVSVM);
        catch
            % If M is already a trained model, estimate training loss via CV on XtrainS
            try
                loss = resubLoss(M); % fallback
            catch
                loss = NaN;
            end
        end

        fprintf('Kernel %s finished. CV loss: %.4f\n', kf, loss);

        % Select best model (lower loss)
        if ~isnan(loss) && (loss < bestLoss)
            bestLoss = loss;
            bestModel = M;
            bestKernel = kf;
        end
    catch ME
        warning('Optimization failed for kernel %s: %s', kf, ME.message);
        continue;
    end
end

if isempty(bestModel)
    error('No optimized model produced across kernels. Check warnings above.');
end

fprintf('\nSelected best kernel: %s  (CV loss = %.4f)\n', upper(bestKernel), bestLoss);

%% ---------- Prepare final trained SVM model for prediction ----------
% If bestModel is an Optimization output containing Trained models cell, extract
trainedModelToUse = bestModel;
if isprop(bestModel,'Trained') && ~isempty(bestModel.Trained)
    % BestModel may be an OptimizationResults object; take the first trained learner
    if isa(bestModel.Trained{1}, 'ClassificationSVM')
        trainedModelToUse = bestModel.Trained{1};
    else
        % attempt to get CompactClassificationSVM
        trainedModelToUse = bestModel.Trained{1};
    end
end

% Ensure we have a usable SVM object
if ~isa(trainedModelToUse, 'ClassificationSVM') && ~isa(trainedModelToUse, 'CompactClassificationSVM')
    % try to get from bestModel if available
    try
        trainedModelToUse = compact(bestModel);
    catch
        % as last resort, error
        error('Could not extract a trained ClassificationSVM object from the optimization result.');
    end
end

%% ---------- Evaluate on held-out test set ----------
try
    Ypred = predict(trainedModelToUse, XtestS);
catch ME
    % sometimes optimized object stores model differently — try fallback
    if isprop(bestModel,'Trained') && ~isempty(bestModel.Trained)
        trained = bestModel.Trained{1};
        Ypred = predict(trained, XtestS);
        trainedModelToUse = trained;
    else
        rethrow(ME);
    end
end

testAcc = mean(Ypred == Ytest) * 100;
fprintf('Test accuracy: %.2f%%\n', testAcc);

figure;
cm = confusionchart(Ytest, Ypred);
cm.Title = sprintf('Test Confusion Matrix (Kernel: %s)', upper(bestKernel));
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ---------- Save model and normalization ----------
try
    TrainedModel = trainedModelToUse; %#ok<NASGU>
    mu_save = mu; sigma_save = sigma; bestKernel_save = bestKernel; bestLoss_save = bestLoss; %#ok<NASGU>
    save(saveFile, 'TrainedModel', 'mu_save', 'sigma_save', 'bestKernel_save', 'bestLoss_save', '-v7.3');
    fprintf('Saved optimized model to: %s\n', saveFile);
catch ME
    warning('Failed to save optimized model cleanly: %s', ME.message);
    save(saveFile, 'bestModel', 'mu', 'sigma', 'bestKernel', 'bestLoss', '-v7.3');
    fprintf('Saved fallback variables to: %s\n', saveFile);
end

fprintf('Done.\n');

end
