function [E, c, v, A, a]=circumellipsoid(X, tol)
%CIRCUMELLIPSOID Return ellipsoid circumscribing set of point X (in rows).
% Computes an approximate minimum volume encapsulating ellipsoid (MVEE).
%
% Syntax:
%     [E, c]=circumellipsoid(X, tol)
%     [____, v]=circumellipsoid(___)
%     [_______, A, a]=circumellipsoid(___)
%
% Inputs:
%     X = n x d matrix of n d-dimensional points
%     tol = MVEE approximation tolerance (default: 0.005)
%
% Outputs:
%     E = symmetric positive definite matrix describing the ellipsoid, i.e.
%         (x-c)' E (x-c) = 1
%     c = center of the ellipsoid.
%     v = volume of the ellipsoid
%     A = ellipsoid principle axes
%     a = lengths of ellipsoid principle axes
%
    if nargin < 2
        tol = 0.005;
    end
    [E, c] = mvee(X', tol);
    if nargout > 2
        d = size(X,2);
        [a, A] = principaxesellip(E, d);
        v = volellip(d, a);
        if ~isfinite(v) || (v == 0)
            error('CIRCUMELLIPSOID:SingularMVEE', 'Infinite or zero volume MVEE.');
        end
    end
end
