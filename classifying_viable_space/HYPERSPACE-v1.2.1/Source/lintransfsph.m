function [ T, d ] = lintransfsph( E,d, s )
%LINTRANSFSPH Linear transformation of a d-dim unit ball to the d-dim
% ellipsoid E, both centered at the origin.
%
% With an optionl scaling s of the ellispoids axes (by default 1). For a
% scalar s, we have x'*E*x <= s^2.
%

    if nargin < 3
        s = 1;
    end
    % W is the axes orientation (rotation matrix), d is the length of each axis
    [d, W] = principaxesellip(E,d);
    % scale axes
    d = s .* d;
    T = W * diag(d) * W';
end
