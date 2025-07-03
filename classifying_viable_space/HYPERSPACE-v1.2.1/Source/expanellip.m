function [V, converged, E,c,v, n_shell] = expanellip(funcion, V0,d, nf_max,n0, bmin,bmax, E_vec,c_vec, grate, varargin)
%EXPANELLIP Expand ellipsoid around V0 until convergence.
%
%
% Syntax:
%     V = expanellip(funcion, V0,d, nf_max,n0, bmin,bmax, E_vec,c_vec, g0,grate)
%     [_, converged, E,c,v, n] = expanellip(___)
%     [___] = expanellip(___, tol, max_niter)
%
%
% Inputs:
%     funcion = indicator function to check if a given point is viable
%
%     V0 = set of viable points to expand ellipsoid around
%     d = dimension of the points
%
%     nf_max = maximum number of function evaluations
%     n0 = number of points sampled in the initial ellipsoid expansion,
%          done around MVEE(V0)
%
%     bmin = vector with the lower bounds for samples
%     bmax = vector with the upper bounds for samples
%
%     E_vec = matrices describing ellipsoids to avoid
%     c_vec = centers of the ellipsoids
%
%     grate = volume growth rate for the ellipsoid expansions (number >1)
%
%     tol = tolerance to accept the convergence of the expansions volume
%           (default: 0.05)
%     max_niter = maximum number of expansion attempts (default: 1e3)
%
% Outputs:
%     V = set of expanded viable points, including V0
%
%     converged = bool flag indicating if the expansion has converged
%
%     E = symmetric positive definite matrix describing the ellipsoid, i.e.
%         (x-c)' E (x-c) = 1
%     c = center of the ellipsoid
%     v = volume of the ellipsoid
%
%     n = final shell sampling resolution (>= n0)
%
debugging = isdebugging;


assert(size(V0,1) > d, 'Not enough initial points to expand the ellipsoid around.');
V=V0; % Viable points found

if numel(varargin) < 1
    tol=0.05;
else
    tol=varargin{1};
end
if numel(varargin) < 2
    max_niter=1e3;
else
    max_niter=round(varargin{2});
end

exp_niter=min(max_niter, ceil(nf_max/(2*n0))); % expected nr of iterations
min_niter=min(ceil(exp_niter/4), max_niter); % minimum nr of iterations
assert(exp_niter >= min_niter && exp_niter <= max_niter);

vol_iter=zeros(1,max_niter+1); % ellipsoid volume after each expansion
conv_niter=max(2,min_niter-1); % check the volume convergence wrt so many
                               % previous expansions
conv_nfr_min = 0.1; % but check the convergence only if ratio of number of
                    % fevals per available fevals (nfr) is higher than that
converged=false;


% Vol. of the ellipsoid with axes of length `a`, multiplied by a factor
% of `gm` is equal to:
%     gm^d * volellip(d, a)
% To grow volume by gr by stretching ellipsoids axes equally, each axis
% has to multiplied by:
%     gm = gr^(1/d)
%assert(numel(grate) == 1);
ga0 = ones(1, d); % no scalling
ga_mul = power(grate, 1/d); % each axis scalling
ga = ga_mul * ga0; % axes scalling in each expansion
%assert(numel(ga) == d);


n_shell=n0; % Nr of samples to take in the next expansion
nf_left = nf_max; % Nr of left tests (function evaluations)
nf_shell = 0;  % Nr of points tested in the last expansion
niter = 0; % expansion (iteration) number

% initial ellipsoid to expand
[E, c, v] = circumellipsoid(V);
vol_iter(niter+1) = v;

next_shell = (n_shell > 0) && (nf_left > 0) && (niter < max_niter);
while next_shell % nf_left > 0
    niter = niter + 1;

    %%%% Sample expansion
    % adapt the sampling resolution to keep the nf_shell approx. at n0
    % nspt: nr of actual samples taken per a tested sample (>= 1)
    if nf_shell == 0
        % == 0 if init or can't find sample in bounds while not in one of
        % the ellipsoids to avoid
        nspt = 1;
    else
        nspt = n0/nf_shell;
    end
    n_max_next = round(nf_left*nspt);
    n_shell = round(n_shell*nspt);

    B0 = randellipsoidshell(min(n_shell, n_max_next), E,c,d, ga);
    n_shell_act = size(B0,1);

    %%%% Filter samples for testing
    % filter out points of bounds
    B = B0(~isoutofbound(B0,bmin,bmax),:);
    % filter out points overlaping w/ the other ellipsoids
    for iellip=1:size(E_vec,3)
        B=ellipdiff(B,E_vec(:,:,iellip),c_vec(:,:,iellip),d);
    end
    % limit nr of points to the available number of function evaluations
    if size(B,1) > nf_left
        B = randrows(B, nf_left);
    end

    %%%% Test samples (find viable points)
    nf_shell = size(B,1);
    nf_left = max(0, nf_left - nf_shell);
    B1 = B(logical(rowfeval(funcion, 1, B)), :);
    nv_shell = size(B1,1); % nr of viable points found in the last shell

    %%%% Update
    V = [V; B1];

    % re-compute MVEE if found at least one viable point
    if nv_shell > 0
        [E, c, v] = circumellipsoid(V);
    end
    vol_iter(niter+1) = v;

    if debugging
        fprintf('[DEBUG] expanellip, post-iteration #%d: n=%d, vol=%.3g; #viable=%d, #tested=%d, #sampled=%d, #tests left=%d.\n',...
            niter,n_shell,vol_iter(niter+1),nv_shell,nf_shell,n_shell_act,nf_left);
    end

    %%%% Check convergence
    % note: by relative volumes changes of ellipsoids around viable points;
    %       if we don't hit any new viable points, volume will not change
    % note: miniter >= checkiter
    converged = ((1-conv_nfr_min) >= nf_left/nf_max) && (niter >= conv_niter) && conver(vol_iter(1:(niter+1)),conv_niter+1,tol);
    % was: (g(niter+1)<1.2) || ... && conver(g(1:niter),checkiter,g_tol) && ...

    next_shell = ~converged && (nf_left > 0) && (niter < max_niter);
end


if ~converged && (nf_max > 0) % no warning for an empty run (niter==0)
    warning('EXPANELLIP:MaxIter','expanellip failed to converge in %d iterations.',niter);
end
if debugging % report finish
    fprintf('[DEBUG] expanellip, finished after iteration #%d (max. %d), after %d/%d fevals, with samp. res. n=%d (n0=%d).\n',niter,max_niter,(nf_max-nf_left),nf_max,n_shell,n0);
end


end
