function [ncmax, cmincount]=nclusmax(n, dim, ncmax0)
%NCLUSMAX Given number of points n and their dimenions dim, compute the
% maximal number of clusters (min. 1), together with a minimal count of
% points per cluster.
%
% Initial number of cluster ncmax0 is optional (by default equal to dim).
% Applies pigeonhole principle to lower ncmax0, i.e. do not cluster into more
% clusters than possible.
%
    if (dim < 1)
        error('NCLUSMAX:IllegalArgument','Non-positive dimension.');
    end
    if nargin < 3
        ncmax0 = dim;
        % TODO: make ncmax0 a parameter of the package public methods (ELexp or MCexp w/ conv check)
    elseif (ncmax0 < 1)
        error('NCLUSMAX:IllegalArgument','Non-positive max. nr of clusters.');
    end

    % number of "enough" points per cluster
    cmincount = 2*(dim+1);
    % max nr of clusters
    %ncmax = min(dim*max(1,floor(log10(n))), floor(n / cmincount));
    ncmax = min(ncmax0, max(1, floor(n / cmincount)));
end
