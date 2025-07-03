function [E, c] = mvee(V, tol)
%MVEE Find a minimal volume enclosing ellipsoid (MVEE) for a set of points
% given in _columns_ of V, within given tolerance (tol).
%
% A wrapper for lowner function +/- finding first a convex hull; see:
%    help lowner
%
% Syntax:
%       [E, c]=mvee(V, tol)
%
% Inputs:
%       V = d x n matrix of n d-dimensional points
%       tol = MVEE approximation tolerance
%
% Outputs:
%       E = symmetric positive definite matrix describing the ellipsoid, i.e.:
%           (x-c)' E (x-c) = 1
%       c = center of the ellipsoid.
%

[E, c] = lowner(V, tol);
%Remark: don't use convex hull - slows things down a lot for large dim
%        (size(V,1))
%[E, c]=lowner(V(:,unique(convhulln(V'))), tol);

end
