function [X, v]=randellipsoid(n, A,c,d, varargin)
%RANDELLIPSOID Get n uniform samples from a d-dimensional ellipsoid A centered
% at a vector c.
%
% Syntax:
%     [X, v]=randellipsoid(n, A,c,d, ...)
%
% Inputs:
%     n = number of sampled points.
%     A = symmetric positive definite matrix describing the ellipsoid, i.e.:
%         (x-c)' A (x-c) = 1
%     c = center of the ellipsoid.
%     d = dimension of the ellipsoid
%     ... = optional arguments of lintransfsph, such as scaling of the
%           ellipsoid axes
%
% Outputs:
%      X = n x d sample from the ellipsoid
%      v = volume of the ellipsoid
%

    % Sample a unit d-ball
    B=randsphere(n,d,1);

    % Transform ball samples to the ellipsoid
    [T, a] = lintransfsph(A, d, varargin{:});
    X = B * T';

    % Center the B set in the center of the ellipsoid
    c_row = c(:)';
    X = bsxfun(@plus, X, c_row(1:d));

    if nargout > 1
        % Compute volume of the ellipsoid
        v = volellip(d, a);
    end

end
