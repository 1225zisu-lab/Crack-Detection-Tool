function crackDetectionApp_v2()
% Crack Detection & Measurement Tool — by Om Ray

clc;

%% ============================================================
%   MODEL + FEATURE EXTRACTION LOADING
% ============================================================
modelPath = fullfile(pwd,"results","models","SVM_crack_detector_v1.mat");
featFcn   = [];
SVMmodel  = [];
mu        = [];
sigma     = [];

if exist(modelPath,"file")==2
    S = load(modelPath);
    fn = fieldnames(S);
    for k=1:numel(fn)
        v = S.(fn{k});
        if isa(v,"ClassificationSVM") || isa(v,"CompactClassificationSVM")
            SVMmodel = v;
            break;
        end
    end
    if isempty(SVMmodel) && isfield(S,"TrainedModel")
        SVMmodel = S.TrainedModel;
    end

    if isfield(S,"mu_save") && isfield(S,"sigma_save")
        mu = S.mu_save;
        sigma = S.sigma_save;
    end
end

if exist("extractCrackFeatures","file")==2
    featFcn = @extractCrackFeatures;
elseif exist("extractFeatures","file")==2
    featFcn = @extractFeatures;
end

if isempty(featFcn)
    uialert(uifigure(),"No extractFeatures() found","Error");
    return;
end

%% ============================================================
%   DEFAULT STATE
% ============================================================
app.I = [];
app.mask = [];
app.lastFile = "";
app.pixelToMM = 0.1;

app.model = SVMmodel;
app.mu = mu;
app.sigma = sigma;
app.featFcn = featFcn;

%% ============================================================
%   THEME
% ============================================================
bg = [0.96 0.96 0.97];
panelBG = [0.92 0.92 0.94];
textC = [0.1 0.1 0.1];
blue = [0 0.45 0.8];
green = [0.2 0.65 0.2];
orange = [0.85 0.55 0.15];

%% ============================================================
%   MAIN GUI
% ============================================================
fig = uifigure("Name","Crack Detection & Measurement Tool", ...
               "Color",bg, ...
               "Position",[150 120 1180 720]);

uilabel(fig,"Text","CRACK DETECTION & MEASUREMENT TOOL", ...
    "FontSize",18,"FontWeight","bold","FontColor",textC, ...
    "Position",[20 680 600 30]);
uilabel(fig,"Text","by Om Ray","FontAngle","italic", ...
    "FontColor",textC,"Position",[20 660 200 20]);

btnLoad   = uibutton(fig,"push","Text","Load Image", ...
    "Position",[740 680 110 30],"BackgroundColor",blue,"FontColor","w");
btnDetect = uibutton(fig,"push","Text","Detect (SVM)", ...
    "Position",[860 680 110 30],"BackgroundColor",blue,"FontColor","w", ...
    "Enable","off");
btnAuto   = uibutton(fig,"push","Text","Auto Mask & Measure", ...
    "Position",[980 680 150 30],"BackgroundColor",green,"FontColor","w", ...
    "Enable","off");

ax = uiaxes(fig,"Position",[20 90 760 560],"BackgroundColor",bg);
axis(ax,"off");
title(ax,"No image loaded","Color",textC);

right = uipanel(fig,"Title","Controls & Results", ...
    "BackgroundColor",panelBG,"Position",[800 90 360 560]);

lblStat = uilabel(right,"Text","Status: Idle", ...
    "Position",[12 500 330 22],"FontColor",textC);
lblPred = uilabel(right,"Text","Prediction: -", ...
    "Position",[12 470 330 22],"FontWeight","bold","FontColor",textC);
lblConf = uilabel(right,"Text","Confidence: -", ...
    "Position",[12 445 330 22],"FontColor",textC);

uilabel(right,"Text","Calibration (mm/pixel):", ...
    "Position",[12 410 200 22],"FontColor",textC);
editCal = uieditfield(right,"numeric","Value",0.1, ...
    "Position",[12 385 140 28]);

btnCal = uibutton(right,"push","Text","Calibrate (2 clicks)", ...
    "Position",[170 385 170 28],"BackgroundColor",blue,"FontColor","w");

btnManual = uibutton(right,"push","Text","Manual Measure", ...
    "Position",[12 345 160 30],"Enable","off","BackgroundColor",blue,"FontColor","w");
btnClr    = uibutton(right,"push","Text","Clear", ...
    "Position",[182 345 158 30],"Enable","off");

lblLen = uilabel(right,"Text","Length: - mm", ...
    "Position",[12 310 330 22],"FontWeight","bold","FontColor",textC);
lblWid = uilabel(right,"Text","Width: - mm", ...
    "Position",[12 285 330 22],"FontColor",textC);

btnSaveImg = uibutton(right,"push","Text","Save Annotated", ...
    "Position",[12 245 160 30],"Enable","off","BackgroundColor",blue,"FontColor","w");

btnPDF = uibutton(right,"push","Text","Export PDF", ...
    "Position",[182 245 158 30], ...
    "Enable","off","BackgroundColor",orange,"FontColor","w");

uilabel(right,"Text","Loaded file:", ...
    "Position",[12 210 330 22],"FontColor",textC);
txtFile = uilabel(right,"Text","-", ...
    "Position",[12 185 330 20],"FontColor",textC,"Interpreter","none");

msgBox = uitextarea(right,"Position",[12 10 336 170], ...
    "Editable","off","Value",["Messages..."]);

%% store label refs
app.lbl.Stat = lblStat;
app.lbl.Pred = lblPred;
app.lbl.Conf = lblConf;
app.lbl.Len  = lblLen;
app.lbl.Wid  = lblWid;
app.lbl.File = txtFile;
app.lbl.Log  = msgBox;
app.ax       = ax;
app.fig      = fig;

%% enable SVM detect
if ~isempty(app.model)
    btnDetect.Enable = "on";
end

%% callbacks
btnLoad.ButtonPushedFcn   = @(~,~) onLoad();
btnDetect.ButtonPushedFcn = @(~,~) onDetect();
btnAuto.ButtonPushedFcn   = @(~,~) onAuto();
btnManual.ButtonPushedFcn = @(~,~) onManual();
btnClr.ButtonPushedFcn    = @(~,~) onClr();
btnCal.ButtonPushedFcn    = @(~,~) onCal();
btnPDF.ButtonPushedFcn    = @(~,~) onPDF();
btnSaveImg.ButtonPushedFcn= @(~,~) onSave();
%% ============================================================
%   UTILITY — LOG
% ============================================================
function log(msg)
    % Ensure msg is a string
    if ~isstring(msg)
        msg = string(msg);
    end

    t = datestr(now,'yyyy-mm-dd HH:MM:SS');
    line = t + "  " + msg;

    v = app.lbl.Log.Value;

    % Ensure v is a column string array
    if ischar(v)
        v = string(v);
    elseif iscell(v)
        v = string(v(:));
    elseif isstring(v)
        v = v(:);
    else
        v = string(v);
    end

    app.lbl.Log.Value = [line; v];
end

%% ============================================================
%   LOAD IMAGE
% ============================================================
function onLoad()
    [f,p] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif'});
    if isequal(f,0), return; end

    fp = fullfile(p,f);

    try
        I = imread(fp);
    catch
        uialert(app.fig,"Could not load image","Error");
        return;
    end

    app.I = I;
    app.lastFile = fp;
    app.mask = [];

    cla(app.ax);
    imshow(I,'Parent',app.ax);
    title(app.ax,f,"Color",[0 0 0]);
    app.lbl.File.Text = fp;

    app.lbl.Stat.Text = "Status: image loaded";
    app.lbl.Pred.Text = "Prediction: -";
    app.lbl.Conf.Text = "Confidence: -";
    app.lbl.Len.Text  = "Length: - mm";
    app.lbl.Wid.Text  = "Width: - mm";

    log("Loaded "+f);

    % enable manual measurement + detecting
    btnManual.Enable = "on";
    btnAuto.Enable   = "on";
    if ~isempty(app.model)
        btnDetect.Enable = "on";
    end
end

%% ============================================================
%   DETECT (SVM)
% ============================================================
function onDetect()
    if isempty(app.I)
        uialert(app.fig,"Load image first","No Image");
        return;
    end

    I = app.I;
    Igray = I; 
    if size(I,3)==3
        Igray = rgb2gray(I);
    end

    f = imresize(Igray,[200 200]);
    fv = app.featFcn(f);

    if ~isempty(app.mu)
        fv = (fv - app.mu) ./ app.sigma;
    end

    try
        [label,score] = predict(app.model,fv);
    catch
        uialert(app.fig,"Prediction failed","Error");
        return;
    end

    conf = max(score);

    app.lbl.Pred.Text = "Prediction: "+string(label);
    app.lbl.Conf.Text = sprintf("Confidence: %.1f%%",conf*100);
    app.lbl.Stat.Text = "Status: Detection done";

    log("Detected "+string(label));

    % auto-mask for view only
    BW = createMask(Igray);
    app.mask = imresize(BW,[size(I,1) size(I,2)]);

    showOverlay(label,conf);

    btnSaveImg.Enable = "on";
    btnPDF.Enable     = "on";
end

%% ============================================================
%   MASK CREATION
% ============================================================
function BW = createMask(Igray)
    BW = imbinarize(Igray,'adaptive','Sensitivity',0.45);
    BW = medfilt2(BW,[3 3]);
    BW = imopen(BW,strel('disk',2));
    BW = bwareaopen(BW,40);
    E  = edge(Igray,'Canny');
    BW = BW & imdilate(E,strel('line',3,0));
    BW = bwareaopen(BW,20);
end

%% ============================================================
%   SHOW OVERLAY
% ============================================================
function showOverlay(label,conf)
    Iorig = app.I;
    mask  = app.mask;

    cla(app.ax);
    imshow(Iorig,'Parent',app.ax);
    hold(app.ax,"on");

    if ~isempty(mask)
        col = cat(3,zeros(size(mask)),255*ones(size(mask)),zeros(size(mask)));
        hm = imshow(uint8(col),'Parent',app.ax);
        set(hm,'AlphaData',0.32*mask);

        CC = bwconncomp(mask);
        if CC.NumObjects>0
            S = regionprops(mask,'BoundingBox');
            bb = S(1).BoundingBox;
            rectangle(app.ax,"Position",bb, ...
                "EdgeColor",[1 0.35 0.35],"LineWidth",1.6);
            text(app.ax,bb(1),bb(2)-10, ...
                sprintf("%s | %.1f%%",label,conf*100), ...
                "Color","y","FontWeight","bold");
        end
    end
    hold(app.ax,"off");
end
%% ============================================================
%   AUTO-MASK + MEASURE
% ============================================================
function onAutoMask()
    if isempty(app.I)
        uialert(app.fig,"Load image first","No Image");
        return;
    end

    I = app.I;
    Igray = I; 
    if size(I,3)==3
        Igray = rgb2gray(I);
    end

    BW = createMask(Igray);
    if isempty(BW)
        uialert(app.fig,"No crack detected","Info");
        return;
    end

    % Resize to 200x200 for skeleton
    BW2 = imresize(BW,[200 200]);

    % Length calc
    sk = bwmorph(BW2,'skel',inf);
    pxLen = sum(sk(:));
    lenMM = pxLen * app.pixelToMM;

    % Width estimation
    D = bwdist(~BW2);
    skIdx = find(sk);
    if isempty(skIdx)
        widthMM = 0;
    else
        rad = mean(D(skIdx));
        widthMM = 2*rad*app.pixelToMM;
    end

    app.mask = imresize(BW2,[size(I,1) size(I,2)]);

    app.lbl.Len.Text = sprintf("Length: %.2f mm",lenMM);
    app.lbl.Wid.Text = sprintf("Width: %.2f mm",widthMM);

    showOverlay("Auto",0);
    log("Auto measurement");
    btnSaveImg.Enable = "on";
    btnPDF.Enable     = "on";
end


%% ============================================================
%   MANUAL 2-POINT MEASUREMENT
% ============================================================
function onManual()
    if isempty(app.I)
        uialert(app.fig,"Load image first","No image");
        return;
    end

    title(app.ax,"Click two points");

    [x,y] = ginput(2);
    if numel(x)<2
        return;
    end

    px = sqrt((x(2)-x(1))^2 + (y(2)-y(1))^2);
    lenMM = px * app.pixelToMM;

    imshow(app.I,"Parent",app.ax);
    hold(app.ax,"on");
    plot(app.ax,x,y,"-r","LineWidth",2);
    plot(app.ax,x,y,"or","MarkerFaceColor","r");
    hold(app.ax,"off");

    app.lbl.Len.Text = sprintf("Length: %.2f mm",lenMM);
    log("Manual length "+lenMM);
end


%% ============================================================
%   CALIBRATION (2-POINT)
% ============================================================
function onCal2()
    if isempty(app.I)
        uialert(app.fig,"Load image first","No image");
        return;
    end

    title(app.ax,"Click two points w/ known distance");

    [x,y] = ginput(2);
    if numel(x)<2
        return;
    end

    px = sqrt((x(2)-x(1))^2 + (y(2)-y(1))^2);

    answer = inputdlg("Enter real-world distance (mm):","Calibration",1,{"10"});
    if isempty(answer), return; end

    mm = str2double(answer{1});
    if isnan(mm) || mm<=0
        uialert(app.fig,"Invalid distance","Error");
        return;
    end

    app.pixelToMM = mm / px;
    app.lbl.Stat.Text = sprintf("Cal: %.5f mm/px",app.pixelToMM);
    log("Calibration done "+app.pixelToMM);
end


%% ============================================================
%   SAVE ANNOTATED IMAGE
% ============================================================
function onSaveImg()
    if isempty(app.I)
        uialert(app.fig,"Load image first","No image");
        return;
    end

    [f,p] = uiputfile("*.png","Save annotated image");
    if isequal(f,0), return; end

    F = figure("Visible","off");
    imshow(app.I);
    hold on;
    if ~isempty(app.mask)
        col = cat(3,zeros(size(app.mask)),255*ones(size(app.mask)),zeros(size(app.mask)));
        h = imshow(uint8(col));
        set(h,"AlphaData",0.35*app.mask);
    end

    T = app.lbl.Len.Text;
    if ~strcmp(T,"Length: - mm")
        text(10,10,T,"Color","yellow","FontSize",14,"FontWeight","bold");
    end

    fr = getframe(gca);
    imwrite(fr.cdata, fullfile(p,f));
    close(F);

    app.lbl.Stat.Text = "Annotated saved";
    log("Saved annotated");
end
%% ============================================================
%   EXPORT PDF
% ============================================================
function onPDF()
    if isempty(app.I)
        uialert(app.fig,"Load + Detect first","No image");
        return;
    end

    dt = datestr(now,"dd-mm-yyyy");
    name = "Test_Results_"+dt+".pdf";
    outDir = fullfile(pwd,"reports");
    if ~exist(outDir,"dir"), mkdir(outDir); end
    outFile = fullfile(outDir,name);

    F = figure("Visible","off","Color","w","Position",[200 80 900 1100]);

    % Title
    annotation(F,"textbox",[0 0.93 1 0.06],...
        "String","STRUCTURAL CRACK ANALYSIS & MEASUREMENT REPORT", ...
        "FontSize",18,"FontWeight","bold","HorizontalAlignment","center", ...
        "EdgeColor","none");

    annotation(F,"textbox",[0 0.89 1 0.04],...
        "String","Report generated using CrackVision — by Om Ray", ...
        "FontSize",11,"FontAngle","italic","HorizontalAlignment","center", ...
        "EdgeColor","none");

    % Separator
    annotation(F,"line",[0.05 0.95],[0.875 0.875],'Color',[0.7 0.7 0.7]);

    % Left: Original + mask
    ax1 = axes("Parent",F,"Position",[0.07 0.45 0.42 0.43]);
    imshow(app.I,"Parent",ax1);
    title(ax1,"Original Image","FontWeight","bold");

    if ~isempty(app.mask)
        hold(ax1,"on");
        col = cat(3,zeros(size(app.mask)),255*ones(size(app.mask)),zeros(size(app.mask)));
        h = imshow(uint8(col),"Parent",ax1);
        set(h,"AlphaData",0.28*app.mask);
        hold(ax1,"off");
    end

    % Right block info
    ax2 = axes("Parent",F,"Position",[0.55 0.48 0.40 0.40]);
    axis(ax2,"off");

    text(ax2,0,0.92,"FILE DETAILS:","FontSize",12,"FontWeight","bold");
    text(ax2,0,0.82,"File: "+ app.lastFile,"Interpreter","none");
    text(ax2,0,0.70,app.lbl.Pred.Text,"FontSize",12,"FontWeight","bold");
    text(ax2,0,0.62,app.lbl.Conf.Text);
    text(ax2,0,0.50,app.lbl.Len.Text);
    text(ax2,0,0.42,app.lbl.Wid.Text);
    text(ax2,0,0.30,sprintf("Calibration: %.6f mm/px",app.pixelToMM));
    text(ax2,0,0.18,"Generated: "+datestr(now,"dddd, dd mmm yyyy"));

    % Messages
    ax3 = axes("Parent",F,"Position",[0.07 0.18 0.88 0.22]);
    axis(ax3,"off");
    text(ax3,0,0.95,"Remarks:","FontWeight","bold","FontSize",12);
    msgs = app.lbl.Log.Value;
    if ~isempty(msgs)
        txt = msgs(1);
        text(ax3,0,0.80,txt,"Interpreter","none");
    end

    % Footer watermark
    annotation(F,"textbox",[0.08 0.04 0.84 0.12], ...
        "String","Crack Inspection & Analysis Tool — by Om Ray", ...
        "HorizontalAlignment","center","FontSize",36,"FontWeight","bold", ...
        "Color",[0.92 0.93 0.96], "EdgeColor","none");

    text(0.5,0.03,"Generated with CrackDetectionApp v2.0   © 2025", ...
        "Units","normalized","HorizontalAlignment","center","FontSize",9);

    try
        exportgraphics(F,outFile,"ContentType","vector");
        close(F);
        uialert(app.fig,"PDF saved to reports folder","PDF Saved");
        log("PDF saved: "+outFile);

        % Auto-open
        try
            if ispc, winopen(outFile); 
            elseif ismac, system("open "+outFile); 
            else, system("xdg-open "+outFile); 
            end
        catch
        end

    catch ME
        close(F);
        uialert(app.fig,"PDF export failed: "+ME.message,"PDF Error");
    end
end


%% ============================================================
%   BATCH TEST
% ============================================================
function onBatch()
    folder = uigetdir(pwd,"Select folder with images");
    if isequal(folder,0), return; end

    files = [dir(fullfile(folder,"*.jpg")); ...
             dir(fullfile(folder,"*.png")); ...
             dir(fullfile(folder,"*.jpeg"))];

    if isempty(files)
        uialert(app.fig,"No images found","Empty");
        return;
    end

    N = numel(files);
    out = strings(N,2);

    for i = 1:N
        try
            I = imread(fullfile(files(i).folder,files(i).name));
            Igray = I;
            if size(I,3)==3
                Igray = rgb2gray(I);
            end
            fv = app.feat(imresize(Igray,[200 200]));
            fvS = (fv - app.mu)./app.sigma;
            lbl = predict(app.model,fvS);

            out(i,1) = files(i).name;
            out(i,2) = string(lbl);

        catch
            out(i,1) = files(i).name;
            out(i,2) = "ERROR";
        end
    end

    assignin("base","batchSummary",out);
    uialert(app.fig,"Batch Results stored in workspace: batchSummary","Batch Done");

    log("Batch finished, "+N+" images");
end


end  % ✅ END OF MAIN FUNCTION
