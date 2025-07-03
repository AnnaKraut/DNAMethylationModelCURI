function [V, E, c, v] = expanellipini(funcion, k0,dim, bmin,bmax, n)
%EXPANELLIPINI Initial cartesian expansion of an ellipsoid around point k0,
% with optional n uniform samples inside of the ellipsoid.
%
% Syntax:
%     V = expanellipini(funcion, k0,dim, bmin,bmax)
%     V = expanellipini(___, n)
%     [V, E, c, v] = expanellipini(___)
%
% Inputs:
%     funcion = indicator function to check if a given point is viable
%
%     k0 = initial viable point to expand ellipsoid around
%     dim = dimension of the points
%
%     bmin = vector with the lower bounds for samples
%     bmax = vector with the upper bounds for samples
%
%     n = number of points sampled in the initial ellipsoid (default: 0)
%
% Outputs:
%     V = set of expanded viable points, including V0
%     E = symmetric positive definite matrix describing the ellipsoid, i.e.:
%         (x-c)' E (x-c) = 1
%     c = center of the ellipsoid.
%     v = volume of the ellipsoid
%

    if nargin < 6, n = 0; end

    % initial cartesian expansion and its ellipsoid
    Vaxes = iterinip(funcion, k0,dim, bmin,bmax);

    if n > 0 || nargout > 1
        [E, c, v] = circumellipsoid(Vaxes, .01);

        % Get an an uniform sample of the initial ellipsoid
        B0 = randellipsoid(n, E, c, dim);
        B = B0(~isoutofbound(B0, bmin, bmax), :);
        Vunif = B(logical(rowfeval(funcion, 1, B)), :);
    else
        Vunif = zeros(0,dim);
    end

    % Collect viable points in the initial ellipsoid
    V = [Vaxes; Vunif];  

end
