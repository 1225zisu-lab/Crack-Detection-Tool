function hu = computeHuMoments(bw)
% computeHuMoments  Compute 7 Hu invariant moments for a binary region image.
%   hu = computeHuMoments(bw)
%   bw: logical or numeric binary image for the region (foreground == 1)
%   hu: 1x7 vector of Hu moments

bw = logical(bw);
[r,c] = size(bw);
[X,Y] = meshgrid(1:c,1:r);
m00 = sum(bw(:));
if m00 == 0
    hu = zeros(1,7);
    return;
end
m10 = sum(sum(X .* bw));
m01 = sum(sum(Y .* bw));
xbar = m10 / m00;
ybar = m01 / m00;

% central moments
mu20 = sum(sum(((X - xbar).^2) .* bw));
mu02 = sum(sum(((Y - ybar).^2) .* bw));
mu11 = sum(sum((X - xbar) .* (Y - ybar) .* bw));
mu30 = sum(sum(((X - xbar).^3) .* bw));
mu03 = sum(sum(((Y - ybar).^3) .* bw));
mu12 = sum(sum(((X - xbar) .* ((Y - ybar).^2)) .* bw));
mu21 = sum(sum((((X - xbar).^2) .* (Y - ybar)) .* bw));

% normalized central moments
eta20 = mu20 / (m00^(1 + (2/2)));
eta02 = mu02 / (m00^(1 + (2/2)));
eta11 = mu11 / (m00^(1 + (2/2)));
eta30 = mu30 / (m00^(1 + (3/2)));
eta03 = mu03 / (m00^(1 + (3/2)));
eta12 = mu12 / (m00^(1 + (3/2)));
eta21 = mu21 / (m00^(1 + (3/2)));

% Hu invariants
hu = zeros(1,7);
hu(1) = eta20 + eta02;
hu(2) = (eta20 - eta02)^2 + 4*(eta11^2);
hu(3) = (eta30 - 3*eta12)^2 + (3*eta21 - eta03)^2;
hu(4) = (eta30 + eta12)^2 + (eta21 + eta03)^2;
hu(5) = (eta30 - 3*eta12)*(eta30 + eta12)*((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) + ...
        (3*eta21 - eta03)*(eta21 + eta03)*(3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
hu(6) = (eta20 - eta02)*((eta30 + eta12)^2 - (eta21 + eta03)^2) + 4*eta11*(eta30 + eta12)*(eta21 + eta03);
hu(7) = (3*eta21 - eta03)*(eta30 + eta12)*((eta30 + eta12)^2 - 3*(eta21 + eta03)^2) - ...
        (eta30 - 3*eta12)*(eta21 + eta03)*(3*(eta30 + eta12)^2 - (eta21 + eta03)^2);
end
