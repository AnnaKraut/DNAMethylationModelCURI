function [W, IDX] = randrows(V,n)
%RANDROWS Return sub-matrix of V of n non-repeating randomly chosen rows
% (n<=size(V,1)).
%
% Syntax:
%       [W] = randrows(V,n)
%
% Inputs:
%     V = matrix
%     n = target number of randomly choosen rows
%
% Outputs:
%     W = matrix with n randomly chosen rows from V
%     IDX = chosen row-indices of V
%

    IDX = randperm(size(V,1),n);
    W = V(IDX, :);

end
