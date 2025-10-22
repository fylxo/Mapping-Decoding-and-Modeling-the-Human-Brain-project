function cmap = redblue(m)
% REDBLUE Red-white-blue colormap for t-maps
%
% Input:
%   m - Number of colors (default: 256)
%
% Output:
%   cmap - Colormap [m x 3]

    if nargin < 1
        m = 256;
    end

    % Create red to white to blue colormap
    n = ceil(m/2);

    % Red to white
    r1 = ones(n, 1);
    g1 = linspace(0, 1, n)';
    b1 = linspace(0, 1, n)';
    lower_half = [r1 g1 b1];

    % White to blue
    r2 = linspace(1, 0, n)';
    g2 = linspace(1, 0, n)';
    b2 = ones(n, 1);
    upper_half = [r2 g2 b2];

    % Combine
    cmap = [lower_half; upper_half(2:end, :)];

    % Adjust to exactly m colors
    if size(cmap, 1) > m
        cmap = cmap(1:m, :);
    end
end
