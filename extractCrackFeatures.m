function feat = extractCrackFeatures(I)
% extractFeatures  Returns a 1x64 feature vector for image I.

%% --- 0) Convert + Resize ---
if size(I,3) == 3
    I = rgb2gray(I);
end
I = im2uint8(I);
I = imresize(I, [200 200]);

feat = [];   % start empty feature vector

%% --- 1) GLCM Texture (4 features) ---
offsets = [0 1; -1 1; -1 0; -1 -1];
glcm = graycomatrix(I, 'Offset', offsets, 'Symmetric', true, 'NumLevels', 16);
stats = graycoprops(glcm);

gl = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
gl = mean(gl, 1);            % average across offsets -> 4 features
feat = [feat, gl];

%% --- 2) LBP (32 features) ---
try
    lbp = extractLBPFeatures(I, 'Upright', false);
    lbp = lbp(1:min(32, numel(lbp)));  % if fewer values exist
    if numel(lbp) < 32
        lbp = [lbp, zeros(1, 32-numel(lbp))];
    end
catch
    lbp = zeros(1,32);
end
feat = [feat, lbp];

%% --- 3) Edge Density (1 feature) ---
edges = edge(I, 'Canny');
edgeDensity = sum(edges(:)) / numel(edges);
feat = [feat, edgeDensity];

%% --- 4) Hu Moments (7 features) ---
BW = imbinarize(I, 'adaptive', 'Sensitivity', 0.45);
BW = imclearborder(BW);
BW = bwareaopen(BW, 30);

CC = bwconncomp(BW);
if CC.NumObjects == 0
    hu = zeros(1,7);
else
    areas = cellfun(@numel, CC.PixelIdxList);
    [~, idx] = max(areas);
    bwReg = false(size(BW));
    bwReg(CC.PixelIdxList{idx}) = true;

    hu = computeHuMoments(bwReg);
    if ~isfinite(hu)
        hu = zeros(1,7);
    end
end
feat = [feat, hu];

%% --- Final: Pad or Trim to 64 ---
feat = feat(:)';   % make row vector

if numel(feat) < 64
    feat = [feat, zeros(1, 64 - numel(feat))];
elseif numel(feat) > 64
    feat = feat(1:64);
end

%% --- Fix: Remove NaN/Inf (MOST IMPORTANT STEP) ---
feat(~isfinite(feat)) = 0;

end
