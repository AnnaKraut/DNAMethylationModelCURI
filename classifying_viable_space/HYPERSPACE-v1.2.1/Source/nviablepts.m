function [U] = nviablepts(viable)
%NVIABLEPTS Total number of viable points
%
% Input:
%   viable (numeric array): a N by D+1 matrix were each of the first U rows is
%                           a non-zero D-dimensional viable point and its cost,
%                           and all the following rows are D-dimensional zero
%                           vectors and an initial cost.
% Output:
%   U (integer): index of the last row containing a non-zero point.
%
    [U,~] = find(viable(:,1:(end-1)),1,'last');
    if isempty(U)
        U = 0;
    end
end
