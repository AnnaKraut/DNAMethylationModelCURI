function [X, n0, v, Xin]=randellipsoidshell(n, A,c,d, g)
%RANDELLIPSOIDSHELL Get ca. n uniform samples from a d-dim ellipsoid shell
% around a d-dim ellipsoid A centered at a point c. Size of the shell is
% obtained by scaling axes of A by a vector g (g>1).
%
% The actual number of samples is estimated as n0=n+ceil(n/(prod(g)-1)) (or
% g^d for a scalar g), i.e. corrected by the number of expected rejected
% samples, which is computed from the ratio of the volume of the inner
% ellipsoid A and the volume of the outer axes-scaled ellipsoid A. On
% average, this gives n target samples.
%
% Syntax:
%       [X, n0, v, Xin]=randellipsoidshell(n, A,c,d, g)
%
% Inputs:
%     n = number of target samples
%     A = symmetric positive definite matrix describing the ellipsoid, i.e.:
%         (x-c)' A (x-c) = 1
%     c = center of the ellipsoid
%     d = dimension of the ellipsoid
%     g = scaling of A axes for the outer shell size
%
% Outputs:
%     X = ca. n d-dim samples from the ellipsoid shell
%     n0 = nr of actually taken samples to get the n samples from the shell
%     v = volume of the outer ellipsoid
%     Xin = the n0-n rejected samples inside the ellipsoid
%
% Examples:
%     A = [2 -1 0; -1 2 -1; 0 -1 2];
%     c = [1 1 1]; d = numel(c);
%     g = 1.25;
%     n = 1e3;
%     [X, n0, v] = randellipsoidshell(n, A,c,d, g);
%     % relative error of nr of target samples
%     abs(n - size(X,1))/n
%     % all samples are in the shell
%     for i=1:size(X,1), r=(X(i,:) - c) * A * (X(i,:) - c)'; assert(r > 1 && r <= g^2); end
%     % nr of actual samples before rejection
%     assert(n+ceil(n/(g^d-1)) == n0)
%     % volume of the inner ellipsoid
%     v0 = volellip(d, principaxesellip(A,d));
%     % relative error of the outer/inner volume ratio
%     abs(v/v0 - g^d)/g^d
%
%     % too much to actualy sample
%     n = round(1.01*nrandmax(d)/g^d);
%     tic; [X, n0, v] = randellipsoidshell(n, A,c,d, g); toc
%     % so actualy sampled less
%     rn0 = n0/(n+ceil(n/(g^d-1)))
%     assert(rn0 < 1)
%     % and got proportionally less
%     abs(rn0 - size(X,1)/n)
%
    assert(all(g>1)); % _outer_ shell
    if numel(g) < d
        if numel(g) < 1
            warning('RANDELLIPSOIDSHELL:PaddingRight','Scaling of axes is a vector of smaller dimension - padding right.');
        end
        g = repmat(g(:)',1,ceil(d/numel(g))); % pad right
        % Note: now numel(g) >= d, so need to trim
    end
    % trim any overhead
    g = g(1:d);
    % Nr of samples required to obtain, on average, n uniform samples from the
    % shell
    n0 = n+ceil(n/(prod(g)-1));
    n0max = nrandmax(d);
    if n0 > n0max
        n0 = n0max;
        warning('RANDELLIPSOIDSHELL:ReducingSampleSize','The actual size of the random sample required to get the target size of the sample is too big - reducing.');
    end
    % Sample the scaled up ellipsoid
    if nargout > 2
        [X0, v] = randellipsoid(n0, A,c,d, g);
    else
        X0 = randellipsoid(n0, A,c,d, g);
    end
    assert(size(X0,1)==n0);
    % Reject samples in the inner ellipsoid
    [X, IX0] = ellipdiff(X0, A,c,d);
    Xin = X0(~IX0,:);
end
