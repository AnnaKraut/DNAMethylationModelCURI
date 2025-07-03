function [Vout, IV] = ellipdiff(V, A,c,d)
%ELLIPDIFF Given a set of points and an ellipsoid description, returns
% points that are outside of the ellipsoid.
%
% Syntax:
%     Vout = ellipdiff(V, A,c,d)
%     [Vout, IV] = ellipdiff(___)
%
% Inputs:
%     V = matrix with the set of points in its rows
%     A = symmetric positive definite matrix describing the ellipsoid, i.e.:
%         (x-c)' A (x-c) = 1
%     c = center of the ellipsoid.
%     d = dimension of the ellipsoid
%
% Outputs:
%     Vout = points in V, which are outside of the ellipsoid
%     IV = logical index vector such that Vout == V(IV,:)
%

    % Shift points to the ellipsoid center, then rotate according to the
    % principle axes of the ellipsoid.
    [D, W] = principaxesellip(A,d);
    c_row = c(:)';
    VA = bsxfun(@minus, V, c_row(1:d)) * W;

    % Compute normalised distance of points to the center of the ellipsoid
    norm_dist = sum(bsxfun(@rdivide, VA, D).^2, 2);

    % Find points that are outside of the ellipsoid
    IV = norm_dist>1;
    Vout=V(IV,:);

end
