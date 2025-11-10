function crackDetectorGUI()
% crackDetectorGUI  Simple GUI for crack detection (disk images + webcam)
% - Expects your saved model at: results/models/SVM_crack_detector_v1.mat
%   containing variables: TrainedModel, mu_save, sigma_save
% - Uses your feature extraction function. Prefers:
%       extractCrackFeatures(I)  (if present)
%   else falls back to:
%       extractFeatures(I)
%
% Run: crackDetectorGUI

% -------------------------------------------------------------------------
% Init / try load model
% -------------------------------------------------------------------------
modelPath = fullfile(pwd,'results','models','SVM_crack_detector_v1.mat');
if ~exist(modelPath,'file')
    error('Model file not found: %s. Run train_model.m first.', modelPath);
end

mdlData = load(modelPath);

if isfield(mdlData,'TrainedModel')
    SVMModel = mdlData.TrainedModel;
elseif isfield(mdlData,'TrainedModelToSave')
    SVMModel = mdlData.TrainedModelToSave;
else
    % try other possible names
    fnames = fieldnames(mdlData);
    found = false;
    for k=1:numel(fnames)
        v = mdlData.(fnames{k});
        if isa(v,'ClassificationSVM') || isa(v,'CompactClassificationSVM')
            SVMModel = v; found = true; break;
        end
    end
    if ~found
        error('No SVM model object found in %s', modelPath);
    end
end

% load mu/sigma names used in training
if isfield(mdlData,'mu_save') && isfield(mdlData,'sigma_save')
    mu = mdlData.mu_save;
    sigma = mdlData.sigma_save;
elseif isfield(mdlData,'mu') && isfield(mdlData,'sigma')
    mu = mdlData.mu;
    sigma = mdlData.sigma;
else
    mu = [];
    sigma = [];
    warning('Normalization (mu/sigma) not found in MAT file. Predictions may be wrong if you do not standardize.');
end

% choose feature function available
if exist('extractCrackFeatures','file') == 2
    featFcn = @extractCrackFeatures;
elseif exist('extractFeatures','file') == 2
    featFcn = @extractFeatures;
else
    error('No feature extraction function found. Provide extractCrackFeatures.m or extractFeatures.m in the path.');
end

% -------------------------------------------------------------------------
% Create UI
% -------------------------------------------------------------------------
fig = uifigure('Name','Crack Detector - Om Ray','Position',[200 120 1100 700]);
tgroup = uitabgroup(fig,'Position',[10 60 1080 630]);

tab1 = uitab(tgroup,'Title','Detect');
tab2 = uitab(tgroup,'Title','Webcam');
tab3 = uitab(tgroup,'Title','Settings');
tab4 = uitab(tgroup,'Title','About');

% Controls on Detect tab
panelLeft = uipanel(tab1,'Position',[10 10 300 600],'Title','Controls');
btnLoad = uibutton(panelLeft,'push','Text','Load Image','Position',[20 540 120 30],'ButtonPushedFcn',@(btn,event) onLoadImage());
btnDetect = uibutton(panelLeft,'push','Text','Detect','Position',[160 540 120 30],'ButtonPushedFcn',@(btn,event) onDetect());
lblMode = uilabel(panelLeft,'Text','Display Mode:','Position',[20 500 120 18]);
ddMode = uidropdown(panelLeft,'Items',{'Simple','Overlay (mask+bbox)'},'Position',[20 470 260 22],'Value','Overlay (mask+bbox)');
lblStatus = uilabel(panelLeft,'Text','Status: Ready','Position',[20 430 260 18]);
lblConf = uilabel(panelLeft,'Text','Confidence: -','Position',[20 400 260 18],'FontWeight','bold');
lblLabel = uilabel(panelLeft,'Text','Label: -','Position',[20 370 260 18],'FontWeight','bold');
btnSaveAnn = uibutton(panelLeft,'push','Text','Save Annotated','Position',[20 330 120 30],'ButtonPushedFcn',@(btn,event) onSaveAnnotated());
btnBatch = uibutton(panelLeft,'push','Text','Batch Test Folder','Position',[160 330 120 30],'ButtonPushedFcn',@(btn,event) onBatchTest());

% Axes for image display
ax = uiaxes(tab1,'Position',[330 20 720 590]);
axis(ax,'off');

% Webcam tab controls
panelLeftW = uipanel(tab2,'Position',[10 10 300 600],'Title','Webcam Controls');
btnStartCam = uibutton(panelLeftW,'push','Text','Start Webcam','Position',[20 540 120 30],'ButtonPushedFcn',@(btn,event) onStartCam());
btnSnap = uibutton(panelLeftW,'push','Text','Capture Frame','Position',[160 540 120 30],'ButtonPushedFcn',@(btn,event) onCaptureFrame(),'Enable','off');
btnStopCam = uibutton(panelLeftW,'push','Text','Stop Webcam','Position',[20 500 260 30],'Enable','off','ButtonPushedFcn',@(btn,event) onStopCam());
lblCamStatus = uilabel(panelLeftW,'Text','Camera: stopped','Position',[20 460 260 18]);

axCam = uiaxes(tab2,'Position',[330 20 720 590]);
axis(axCam,'off');

% Settings tab
uicheck = uicheckbox(tab3,'Text','Show overlay on detect','Position',[20 560 200 22],'Value',true);
uicheck.Value = true;
lblInfo = uilabel(tab3,'Text','Model file:','Position',[20 520 400 18]);
lblModel = uilabel(tab3,'Text',modelPath,'Position',[20 500 800 18],'Interpreter','none');

% About tab
txtAbout = sprintf(['Crack Detection GUI\nDeveloped by Om Ray\n\n' ...
    'Fault detection in part by Om Ray\n\n' ...
    'This GUI uses a trained SVM model to classify images as "crack" or "no_crack".' ...
    '\n\nInstructions:\n - Use Load Image to test a single photo.\n - Use Webcam tab to capture live frames.\n - Switch display mode for overlay or simple label.\n']);
uilabel(tab4,'Text',txtAbout,'Position',[20 20 700 400],'FontSize',12);

% App data store
app.currentImage = [];
app.mask = [];
app.SVMModel = SVMModel;
app.mu = mu;
app.sigma = sigma;
app.featFcn = featFcn;
app.webcamObj = [];
app.camTimer = [];
app.ax = ax;
app.axCam = axCam;
app.fig = fig;

% -------------------------------------------------------------------------
% Callback functions
% -------------------------------------------------------------------------
    function onLoadImage()
        [f,p] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp'}, 'Select image');
        if isequal(f,0), return; end
        I = imread(fullfile(p,f));
        app.currentImage = I;
        app.mask = [];
        cla(app.ax); imshow(I,'Parent',app.ax);
        lblStatus.Text = sprintf('Loaded: %s', f);
        lblLabel.Text = 'Label: -';
        lblConf.Text = 'Confidence: -';
    end

    function onDetect()
        if isempty(app.currentImage)
            uialert(fig,'Load an image first.','No Image');
            return;
        end
        lblStatus.Text = 'Running detection...'; drawnow;
        I = app.currentImage;
        % Preprocess for feature extraction
        Igray = I;
        if size(I,3)==3, Igray = rgb2gray(I); end
        IgraySmall = imresize(Igray,[200 200]);
        % extract features using selected function
        fv = app.featFcn(IgraySmall);
        % standardize if mu/sigma available
        if ~isempty(app.mu) && ~isempty(app.sigma)
            fvS = (fv - app.mu) ./ app.sigma;
        else
            fvS = fv;
        end
        % predict
        try
            [label, score] = predict(app.SVMModel, fvS);
        catch ME
            % try compact/TrainedModel inside structure
            try
                [label, score] = predict(app.SVMModel.Trained{1}, fvS);
            catch
                uialert(fig, sprintf('Prediction failed: %s', ME.message), 'Predict Error');
                return;
            end
        end
        conf = max(score);
        lblLabel.Text = sprintf('Label: %s', string(label));
        lblConf.Text = sprintf('Confidence: %.1f%%', conf*100);
        lblStatus.Text = 'Detection done';
        % Make mask for overlay if user wants
        if strcmp(ddMode.Value,'Overlay (mask+bbox)') && uicheck.Value
            mask = detectionMask(Igray);           % small function below
            app.mask = imresize(mask, [size(I,1) size(I,2)]);
            displayOverlay(I, label, conf);
        else
            cla(app.ax); imshow(I,'Parent',app.ax);
            title(app.ax, sprintf('Pred: %s (%.1f%%)', string(label), conf*100));
        end
    end

    function m = detectionMask(Igray)
        % Simple detection mask using adaptive threshold + morph
        BW = imbinarize(Igray, 'adaptive', 'Sensitivity', 0.45);
        BW = medfilt2(BW,[3 3]);
        BW = imopen(BW, strel('disk',2));
        BW = bwareaopen(BW,40);
        % refine with edges
        E = edge(Igray,'Canny');
        BW = BW & imdilate(E, strel('line',3,0));
        BW = bwareaopen(BW,20);
        m = BW;
    end

    function displayOverlay(I, label, conf)
        cla(app.ax);
        imshow(I,'Parent',app.ax); hold(app.ax,'on');
        if ~isempty(app.mask)
            cmap = cat(3, zeros(size(app.mask)), ones(size(app.mask)), zeros(size(app.mask)));
            him = imshow(uint8(255*cmap),'Parent',app.ax);
            set(him,'AlphaData',0.35 * app.mask);
            % bbox
            CC = bwconncomp(app.mask);
            if CC.NumObjects>0
                stats = regionprops(app.mask,'BoundingBox');
                bb = stats(1).BoundingBox;
                rectangle(app.ax,'Position',bb,'EdgeColor',[1 0.2 0.2],'LineWidth',1.5);
                text(app.ax, bb(1), bb(2)-10, sprintf('%s | %.1f%%', string(label), conf*100), 'Color','y','FontWeight','bold');
            end
        end
        hold(app.ax,'off');
    end

    function onSaveAnnotated()
        if isempty(app.currentImage)
            uialert(fig,'No image to save.','Error');
            return;
        end
        f = uiputfile('annotated.png','Save annotated image as');
        if isequal(f,0), return; end
        % render into temp figure
        tmpFig = figure('Visible','off');
        imshow(app.currentImage); hold on;
        if ~isempty(app.mask)
            himt = imshow(cat(3,zeros(size(app.mask)),uint8(255*app.mask),zeros(size(app.mask))));
            set(himt,'AlphaData',0.35*app.mask);
        end
        frame = getframe(gca);
        imwrite(frame.cdata, fullfile(pwd,f));
        close(tmpFig);
        lblStatus.Text = sprintf('Saved annotated: %s', f);
    end

    function onBatchTest()
        % choose folder and test images inside
        folder = uigetdir(pwd,'Select folder containing images to test (jpg/png)');
        if isequal(folder,0), return; end
        files = [dir(fullfile(folder,'*.jpg')); dir(fullfile(folder,'*.png')); dir(fullfile(folder,'*.jpeg'))];
        if isempty(files)
            uialert(fig,'No images found in folder.','No Files');
            return;
        end
        total = numel(files);
        results = strings(total,2); % name, prediction
        correct = 0; knownLabels = false;
        for k=1:total
            try
                I = imread(fullfile(files(k).folder, files(k).name));
                Igray = I; if size(I,3)==3, Igray = rgb2gray(I); end
                fv = app.featFcn(imresize(Igray,[200 200]));
                if ~isempty(app.mu), fvS = (fv - app.mu) ./ app.sigma; else fvS = fv; end
                [lbl, sc] = predict(app.SVMModel, fvS);
                results(k,1) = files(k).name;
                results(k,2) = string(lbl);
                % If filenames contain label hints (crack or no_crack) compute quick accuracy
                name = lower(files(k).name);
                if contains(name, 'crack')
                    trueLbl = "crack"; knownLabels = true;
                elseif contains(name, 'no_crack') || contains(name, 'nocrack') || contains(name, 'no-crack')
                    trueLbl = "no_crack"; knownLabels = true;
                else
                    trueLbl = "";
                end
                if knownLabels && ~isempty(trueLbl)
                    if string(lbl) == trueLbl, correct = correct + 1; end
                end
            catch
                results(k,1) = files(k).name;
                results(k,2) = "ERROR";
            end
        end
        if knownLabels
            uialert(fig, sprintf('Batch done. Accuracy (inferred from filenames): %.2f%% (%d/%d)', ...
                correct/total*100, correct, total), 'Batch Results');
        else
            uialert(fig, sprintf('Batch done. Predictions saved to workspace variable ''batchResults''.'), 'Batch Results');
        end
        assignin('base','batchResults',results);
    end

    function onStartCam()
        % start webcam preview (if available)
        try
            camList = webcamlist;
        catch
            uialert(fig,'No webcam support available (Image Acquisition Toolbox required) or no webcam installed.','Webcam Error');
            return;
        end
        if isempty(camList)
            uialert(fig,'No webcam detected.', 'Webcam Error');
            return;
        end
        cam = webcam(1); % first camera
        app.webcamObj = cam;
        btnStartCam.Enable = 'off';
        btnStopCam.Enable = 'on';
        btnSnap.Enable = 'on';
        lblCamStatus.Text = 'Camera: running';
        % simple timer to update preview
        app.camTimer = timer('ExecutionMode','fixedSpacing','Period',0.1,'TimerFcn',@(~,~) updateCamPreview());
        start(app.camTimer);
    end

    function updateCamPreview()
        if isempty(app.webcamObj), return; end
        try
            frame = snapshot(app.webcamObj);
            cla(app.axCam); imshow(frame,'Parent',app.axCam);
        catch
            % ignore frame errors
        end
    end

    function onCaptureFrame()
        if isempty(app.webcamObj)
            uialert(fig,'Start webcam first.','No Webcam');
            return;
        end
        I = snapshot(app.webcamObj);
        app.currentImage = I;
        app.mask = [];
        cla(app.ax); imshow(I,'Parent',app.ax);
        % switch to detect tab
        tgroup.SelectedTab = tab1;
        lblStatus.Text = 'Frame captured from webcam';
    end

    function onStopCam()
        if ~isempty(app.camTimer) && isvalid(app.camTimer)
            stop(app.camTimer); delete(app.camTimer); app.camTimer = [];
        end
        if ~isempty(app.webcamObj)
            clear app.webcamObj; app.webcamObj = [];
        end
        btnStartCam.Enable = 'on';
        btnStopCam.Enable = 'off';
        btnSnap.Enable = 'off';
        lblCamStatus.Text = 'Camera: stopped';
        cla(app.axCam);
    end

% -------------------------------------------------------------------------
% tidy up on figure close
% -------------------------------------------------------------------------
fig.CloseRequestFcn = @(src,event) onClose();

    function onClose()
        try
            if ~isempty(app.camTimer) && isvalid(app.camTimer)
                stop(app.camTimer); delete(app.camTimer);
            end
        catch
        end
        delete(fig);
    end

end
