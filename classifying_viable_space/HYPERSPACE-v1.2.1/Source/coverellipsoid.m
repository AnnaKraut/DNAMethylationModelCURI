function [Ev, cv, vv, IDX]=coverellipsoid(V,d)
%COVERELLIPSOID Computer ellipsoids cover for first d dimensions of points given
% in rows of V. Parts to cover are computed using clustering, and covered with
% the minimum volume enclosing ellipsoids (MVEE).
%
% Syntax:
%       [Ev, cv, vv, IDX] = coverellipsoid(V,d)
%
% Inputs:
%     V = set of parameter points
%     d = dimension of the points
%
% Outputs:
%     Ev  = d x d x k matrix; for every of the k found clusters, d x d submatrix
%           defines its MVEE.
%     cv  = d x 1 x k matrix with center of every cluster (and ellipsoid).
%     vv  = k vector with volume of every MVEE.
%     IDX = n vector with an index of a cluster for every point (row) of V
%
debugging = isdebugging;


% drop any extra columns, e.g. cost column
V1 = V(:,1:d);
n = size(V1,1);
nc = nclusmax(n, d);
clustered = false;

while ~clustered && nc > 0
    try
        % cluster
        [nc, IDX]=nclus(V1, nc);

        % compute MVEE for every cluster
        Ev=zeros(d,d,nc);
        cv=zeros(d,1,nc);
        vv=zeros(1,nc);
        for i=1:nc
            [Ev(:,:,i), cv(:,:,i), vv(i)] = circumellipsoid(V1(IDX==i,:), 0.005);
        end
        clustered = true;
    catch ME
        if any(strcmp(ME.identifier,{...
                'MATLAB:svd:matrixWithNaNInf',...
                'KHACHIYAN:NonInvertibleMatrix',...
                'CIRCUMELLIPSOID:SingularMVEE'}))
            nc = nc-1;
            if nc > 0
                warning('COVERELLIPSOID:SingularMVEE',...
                    'Failed to compute MVEE for one of the clusters. Retrying with a lower number of clusters.');
                % Remark: instead of running k-means again, we could also just
                %         re-assign points of the faulty cluster (faster)
            end
        else
            rethrow(ME);
        end
    end
end


if ~clustered
    error('COVERELLIPSOID:SingularMVEE','Failed to compute MVEEs in all clusterization tries.');
end

if debugging
    fprintf('[DEBUG] coverellipsoid, MVEE volumes per cluster = %s.\n',mat2str(vv,3));
end

end
