function [res, err, nib] = intestellip(funcion,nfout, E_vec,c_vec,v_vec,d, n, bmin,bmax)
%INTESTELLIP Monte Carlo estimate of the integral of a function within
% a set defined by the ellipsoids and the [bmin, bmax] bound.
%
% Note: integration in the regions shared by two or more ellipsoids is done
%       only once.
%
% Syntax:
%     intestellip(funcion,nfout, E_vec,c_vec,v_vec,d, n, bmin,bmax)
%
%
% Inputs:
%     funcion = function to be integrated.
%     nfout = number of outputs of the function (integrated separately)
%
%     E_vec  = d x d x k matrix of k ellipsoids desc. by d x d matrices
%     c_vec  = d x 1 x k matrix with center of every ellipsoid.
%     v_vec  = k vector with volume of every ellipsoid.
%     d = dimension of the space
%
%     n = number of sampled points
%     bmin = vector with the lower bound of our parameter space.
%     bmax = vector with the upper bound of our parameter space.
%
%
% Outputs:
%        res = matrix with the results of the integral for every ellipsoid,
%              one column for each ellipsoid
%        err = matrix with the error of the integral for every ellipsoid,
%              one column for each ellipsoid
%        nib = row vector of number of samples within bounds for each
%              ellipsoid (sum(nib) <= n)
%
debugging = isdebugging;

    nc=numel(v_vec);
    res=zeros(nfout,nc);
    err=zeros(nfout,nc);
    nib=zeros(1,nc);


    % Monte Carlo integration for every ellipsoid 

    %number of samples per ellipsoids (npe), proportional to their volume
    npe=splitlrm(n, v_vec);
    if debugging
        fprintf('[DEBUG] intestellip, number of points per ellipsoid = %s (prop to their volumes).\n',mat2str(npe));
    end

    for i=1:nc
        % Uniform sampling in every ellipsoid
        [B, vol] = randellipsoid(npe(i),E_vec(:,:,i),c_vec(:,:,i),d);

        % MC integral estimate for non-overllaping points of the ellipsoids
        Eprev_vec = E_vec(:,:,1:(i-1));
        cprev_vec = c_vec(:,:,1:(i-1));
        [res(:,i), err(:,i), nib(i)] = integrator(funcion,nfout, B,vol,...
            bmax,bmin, Eprev_vec,cprev_vec,d);
    end

    if debugging
        fprintf('[DEBUG] intestellip, res = %s +/- %s.\n',...
            mat2str(res,3),mat2str(err,3));
        % Note: in volume estimation (funcion=='unidad'), if all points of
        %       the 1st ellipsoid are within bounds (npe(1) == nib(1)),
        %       then err(1) == 0
    end

end
