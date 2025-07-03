function [nc, IDX, C, validc]=nclus(V, varargin)
%NCLUS Clusterizes a set of points, finding a suitable number of clusters.
%
% Syntax:
%       [nc, IDX, C, validc] = nclus(V)
%       [___] = nclus(V, ncmax)
%
% Inputs:
%       V = k x d matrix of k d-dimensional points (in rows)
%       ncmax = max. number of clusters (default: nclusmax(n, size(V,1)))
%
% Outputs:
%        nc = number of clusters
%        IDX = vector of k indices from 1:nc, assigning a cluster to every
%              point of V
%        C = centroids of every cluster.
%        validc = flag indicating if clustering is valid (each cluster has
%                 enough points; if it's invalid, nc == 1)
%
debugging = isdebugging;

    n = size(V,1);
    dim = size(V,2);
    [nc_max, mincount] = nclusmax(n, dim, varargin{:});

    nc = nc_max;
    n_iter = 0;
    % while: (max. valid not found) and (not all invalid) and
    %       (did not do more steps than the linear search from ncmax would do)
    % do: cluster and check w/ bisections on nr of clusters
    nc_invalid = nc+1;
    nc_valid = 0;
    if n >= mincount
        while (nc_valid+1 < nc_invalid) && (nc > 0) && (n_iter < (nc_max-nc+1))
            n_iter = n_iter+1;
            if debugging
                fprintf('[DEBUG] nclus, trying %d clusters.\n', nc);
            end

            % Divide the set V into nc clusters
            [IDX, C] = grouprows(V,nc);

            % Check if clusterization is valid, i.e. each cluster has enough points
            [validc, npc] = isvalidgrouping(V, nc, IDX, mincount);

            % Bisect nr of cluster for the next check
            if ~validc
                nc_invalid = nc;
                nc = floor(nc/2);
            else
                nc_valid = nc;
                IDX_valid = IDX;
                C_valid = C;
                npc_valid = npc;
                nc = nc + floor((nc_invalid-nc)/2);
            end
        end

        assert(nc_valid > 0);
        nc = nc_valid;
        IDX = IDX_valid;
        C = C_valid;
        validc = true;
        npc = npc_valid;
    else % n < mincount
    % No valid clusterization only if there are not enough points in total;
    % warn and return a single invalid cluster
        warning('NCLUS:TooFewPoints','Too few points to cluster = %d < 2*(dim+1), with dim = %d.',n,dim);
        nc = 1;
        IDX = ones(n,1);
        C = mean(V,1);
        validc = false;
        npc = n;
    end


    if debugging
        if validc
            validstr = 'valid';
        else
            validstr = 'invalid';
        end
        fprintf('[DEBUG] nclus, %s clustering of %d points into %d clusters, with dim=%d.\n',validstr,n,nc,dim);
        fprintf('[DEBUG] nclus, points per cluster: %s.\n',mat2str(npc));
    end

end



function [IDX, C] = grouprows(V, nc)
    rng('shuffle');
    %[IDX, C]= kmeans(V,nc,'emptyaction','singleton');

    n = size(V,1);
    dim = size(V,2);
    % trick: k-means for max. nk pts + assign remaining pts to centroids;
    %        noticeable jump in running time for randn points occurs
    %        between 1e4 and 1e5 samples
    nk = min(5e2*2*dim,n);

    [Va, IVa] = randrows(V,nk);
    [IDXa, C]= kmeans(Va,nc,'emptyaction','singleton');
    % note: it's not true that ismember(C, V, 'rows')); it shouldn't matter

    % Assign not-clustered points by the distance to the center
    IVb = setdiff(1:n,IVa);
    Vb = V(IVb,:);
    IDXb = dsearchn(C, Vb);

    IDX = [IDXa; IDXb];
    % Attention: restore original order of rows
    IVinv([IVa IVb]) = 1:n;
    IDX = IDX(IVinv);
end



function [validc, npc] = isvalidgrouping(V, nc, IDX, mincount)
    ic=1;
    npc = zeros(1,nc);
    validc = true;
    while validc && (ic<=nc)
        VI = V((IDX==ic),:);
        npc(ic) = size(VI,1);
        validc = (npc(ic)>=mincount);
        ic=ic+1;
    end
end
