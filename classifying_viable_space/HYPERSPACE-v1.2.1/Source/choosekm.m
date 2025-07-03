function index = choosekm(V, K)
%CHOOSEKM From a set of points V (in rows), choose a random point that is
% typically far from all points in a set K (in rows).
%
% Syntax:
%     index = choosekm(V, K)
%
% Inputs:
%     V = n x d matrix of n points to select from
%     K = m x d matrix of m points to avoid
%
% Outputs:
%     index = index of the point chosen from V
%

    if isempty(K)
        [~, index] = randrows(V,1);
        return;
    end

    %%% Compute distances
    [~, dists] = dsearchn(K, V);
    % meank = mean(K,1);
    % dists = arrayfun(@(i) norm(Vtotal(i,:)-meank), 1:size(Vtotal,1));
    maximo=max(dists);
    minimo=min(dists);

    %%% Pick a point at a target, random distance
    %%% (scaled as square root from min_v{d(v,K)} to max_v{d(v,K)})
    rng('shuffle'); % in case of parallelization
    y=sqrt(rand());
    randist=(maximo-minimo)*y+minimo;
    [~, index] = min(abs(dists-randist));

end

