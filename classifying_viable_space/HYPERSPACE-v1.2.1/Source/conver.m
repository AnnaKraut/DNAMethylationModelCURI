function [bool, rels] = conver(x, n, tol)
%CONVER Check if the relative difference between the last n-1 elements of
% a vector x and its last element is less than the given relative tolerance
% tol.
%
% Syntax:
%     bool = conver(x, n, tol)
%
% Inputs:
%     x = vector
%     tol = tolerance
%     n = nr of elements to check
%
% Outputs:
%     bool = true, if all relative differences are smaller than the
%            tolerance, false otherwise
%     rels = vector of n-1 relative differences of x((end-n+1):(end-1)) wrt
%            x(end)
%
    assert(numel(x)>=n, 'Too few elements to check.');
    last = x(end);
    rels = abs(x((end-n+1):(end-1))-last)/last;
    bool = all(rels < tol);
end
