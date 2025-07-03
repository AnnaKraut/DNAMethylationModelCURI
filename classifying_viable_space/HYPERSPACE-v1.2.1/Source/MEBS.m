function [OutV, flags, nfeval, ellip] = MEBS(funcion, V0,dim, nf_max,nexpmin,nint, bmin,bmax, grate, tol)
%MEBS Get viable parameter points through the Multiple Ellipsoids based
% sampling.
%
% Syntax:
%     [V, flags, nfeval] = MEBS(funcion, V0,dim, nf_max,nexpmin,nint, bmin,bmax, grate, tol)
%     [___, ellip] = MEBS(___)
%
% Inputs:
%     funcion = function recieves a parameter point and check its viability
%
%     V0 = set of viable seeds (attention: their viability is not verified)
%     dim = dimension of the parameter space
%
%     nf_max = maximum number of function evaluations
%     nexpmin = min. number of parameter points sampled in each ellipsoid
%               expansion attempt
%     nint = points sampled in order to check the convergence
%
%     bmin = vector with the lower bound of our parameter space.
%     bmax = vector with the upper bound of our parameter space.
%
%     grate = volume growth rate for ellipsoid expansions (>1)
%
%     tol = tolerance to accept the convergence of the total cover volume
%
% Outputs:
%     V = all the viable parameter points found with their cost appended
%         in the last column
%     flags = structure with two fields.
%         flags.vol = volume covered by the enclosing ellipsoids in every
%                     iteration (starting at 0)
%         flags.conv = logical true if the ellipsoid cover expansion has
%                      converged
%     nfeval = number of function evaluations
%     ellip : structure describing final ellipsoids cover
%         ellip.A : (dim)x(dim)x(n elip) axes in cols
%         ellip.c : (dim)x(1)x(n elip) centers in cols
%         ellip.v : (n elip) volumes
%
debugging = isdebugging;


if size(V0,1)==0
    error('MEBS:NoSeed','No initial seed provided.');
end

%%%% Initial cover
[viable0, nfeval0] = feval(funcion,'-g');
U0 = nviablepts(viable0); % nr of viable points we don't care for

V0unique = unique(V0,'rows');
n_seeds=size(V0unique,1); % n_seeds > 0
if n_seeds < size(V0,1)
    warning('MEBS:DuplicateSeeds', 'Removed duplicates among the initial seeds.');
end
if n_seeds < (dim+1) % min. #pts required to describe an ellipsoid
    ik0 = choosekm(V0unique,mean(V0unique,1));
    k0_vec = V0unique(ik0,:); % initial expansion point
    Vtotal = expanellipini(funcion, k0_vec,dim, bmin,bmax);
    Vtotal = [V0unique; Vtotal]; % unique(...,'rows') rather unnecessary here
else
    k0_vec = zeros(0, dim); % no initial expansion points
    Vtotal = V0unique;
end
% E_vec, c_vec: current ellipsoids cover of the viable space
try
    [E_vec, c_vec, vol_vec, IDX] = coverellipsoid(Vtotal,dim);
catch ME
    if strcmp(ME.identifier, 'COVERELLIPSOID:SingularMVEE')
        error('MEBS:SingularCover', 'Couldn''t compute the initial MVEE cover - always contains a singular MVEE.');
    else
        rethrow(ME);
    end
end



%%%% Iterate
nf_left = max(0, nf_max);
next_iter = (nf_left > 0);
niter=1; % nr of iteration over whole cover
exp_niter = 5; % expected number of main loops (i.e. loops over all clusters)
ic=0; % cluster number

voltot_iter=inf(exp_niter,1); % total volume covered after each round of expansions
conv_niter=2; % check the convergence wrt so many previous iterations
%assert(niter_conv>=2, 'Convergence can only be done over at least two iterations.');
converged = false; % convergence of the volume of all expansions

% Prep. seeds for every group of expansion; est. max nr of samples for each
[V_ca, nf_max_vec, nc, E_vec,c_vec,vol_vec] = expanellip_prep_seeds(...
    Vtotal, IDX, E_vec,c_vec,vol_vec, floor(nf_left/exp_niter));
if debugging
    fprintf('[DEBUG] MEBS, pre iter %d, max. fevals per expansion: %s.\n',...
        niter, mat2str(nf_max_vec));
end

while next_iter
    ic=ic+1;

    % TESTME: build-up the cover again each time (starting from the largest
    %         volume cluster) instead of just expanding over regions
    %         non-overlapping with the current cover.

    % other ellipsoids
    Ev = E_vec; Ev(:,:,ic) = [];
    cv = c_vec; cv(:,:,ic) = [];
    vv = vol_vec; vv(ic) = [];

    Vexp = V_ca{ic};

    % filter out seeds overlapping w/ other ellipsoids
    for iellip=1:size(Ev,3)
        Vexp = ellipdiff(Vexp, Ev(:,:,iellip), cv(:,:,iellip), dim);
    end
    % TODO would make sense to re-cluster Vexp, becuse it might have been
    %      "splitted" be removing the overlaps
    if debugging
        fprintf('[DEBUG] MEBS, iter: %d + %d/%d, #seeds: %d/%d.\n',...
            niter-1,ic,nc, size(Vexp,1),size(Vexp,1));
    end

    % expand ellipsoid over current seeds, if there is enough of them
    if size(Vexp,1) > dim
        nf_max_iter = min(nf_left, nf_max_vec(ic));
        [k0_vec, E,c,v, nfeval, ~, Vb] = ...
            expanellip_wrapper(funcion, Vexp,dim, k0_vec, Ev,cv,vv,...
                nf_max_iter,nexpmin, bmin,bmax, grate, tol);
        % update
        nf_left = nf_left - nfeval;
        E_vec(:,:,ic) = E;
        c_vec(:,:,ic) = c;
        vol_vec(ic) = v;

        % add right away the boundary points found in the remaining clusters
        IVb = dsearchn(ellipcentrows(cv), Vb); % c_vec(,,(ic+1):nc) == cv(,,ic:(nc-1))
        for j=(ic+1):nc
            V_ca{j}=[V_ca{j}; Vb(IVb==(j-1),:)];
        end
    end

    next_iter = (nf_left > 0);

    % re-compute the cover and check convergence
    if ~next_iter || (ic==nc)
        viable = feval(funcion,'-g');
        % drop the cost column, and the pre-MEBS viable points
        Vtotal = [V0unique; viable((U0+1):(nviablepts(viable)),1:dim)];
        % Estimate volume while re-computing the cover
        % Rem: was mean est. volume, now it's minimal est. volume over repeats
        [voltot_iter(niter), ~, ~, ~, E_vec,c_vec,vol_vec, IDX] = volestellip(...
            Vtotal,dim, nint, bmin,bmax);

        if debugging
            fprintf('[DEBUG] MEBS, volume covered: %.3g (%.3g%% of the bounding box).\n',...
                voltot_iter(niter), 100*voltot_iter(niter)/prod(bmax - bmin));
            fprintf('[DEBUG] MEBS, #sampled parameters points: %d/%d.\n',...
                nf_max-nf_left,nf_max);
        end

        % converged = (m>niter_conv) && conver(Volexp(1:m),tol,niter_conv+1);
        if (niter>conv_niter) && (nf_left > 0)
            [converged, relVolDelta] = conver(voltot_iter(1:niter),...
                conv_niter+1, tol);
            if debugging
                fprintf('[DEBUG] MEBS, max. relative volume change wrt previous %d checks: %2.1f%%.\n',...
                    conv_niter, 100*max(relVolDelta));
            end
        end

        fprintf('[INFO] MEBS, progress: %2.1f%%.\n', 100*(1 - nf_left/nf_max));

        next_iter = ~converged && (nf_left>0);
        if next_iter
            ic=0;
            niter=niter+1;
            exp_niter = max(1, round(exp_niter * (nf_left/nf_max)));

            [V_ca, nf_max_vec, nc, E_vec,c_vec,vol_vec] = expanellip_prep_seeds(...
                Vtotal, IDX, E_vec,c_vec,vol_vec, floor(nf_left/exp_niter));
            if debugging
                fprintf('[DEBUG] MEBS, pre iter %d, max. fevals per expansion: %s.\n',...
                    niter, mat2str(nf_max_vec));
            end
        end
    end
end


[viable, nfeval] = feval(funcion,'-g');
nfeval = nfeval - nfeval0;
U=nviablepts(viable);
OutV=viable((U0+1):U,:);

flags.vol=voltot_iter(1:niter);
assert(all(isfinite(flags.vol)));
flags.conv = converged;


ellip.A = E_vec;
ellip.c = c_vec;
ellip.v = vol_vec;


if all(flags.conv)
    convStr = 'converged';
else
    convStr = 'did not converge';
end
fprintf('[INFO] MEBS, %s in %.2g iterations; found %d viable samples in %d/%d function evaluations; covered volume of %.3g.\n',...
    convStr, niter-1+(ic/nc), size(OutV,1), nfeval, nf_max, flags.vol(end));

if debugging && (dim == 2) && exist('ellipse', 'file')
    scatterellipse(V0unique, OutV(:,1:dim), OutV(:,end), E_vec, c_vec,'[DEBUG] MEBS');
end

end




function [k0v, E,c,v, nfeval, V, Vbound_out] = ...
        expanellip_wrapper(funcion, V0,d, k0v, Ev,cv,vv,...
            nf_max,nexpmin, bmin,bmax, grate, tol)
debugging = isdebugging;

    % In the worst case we'll be expanding ellipsoid circumscribing only V0
    n_seeds=size(V0,1);
    if n_seeds < d+1
        error('EXPANELLIP_WRAPPER:NotEnoughSeeds','Not enough initial seeds provided.');
    end

%     assert(size(Ev,3) == size(cv,3));

    % record nr of function evaluations in this expansion
    [~, nfeval_pre] = feval(funcion,'-g'); % note: w/o nfeval0
    nfeval = 0;

    %%%% Initial cartesian expansion
    % avoid centers of previous cover elements and previous expansion points
    Vbound = zeros(0,d);
    Vbound_out = zeros(0, d);
    cv_temp = ellipcentrows(cv);
    % find at least d+1 boundary points non-overlapping w/ the other ellip;
    % try at least max. once on avg. from each seed, but no more than
    % nr of fevals (assuming avg. 4 sections in each dim direction)
    expini_max = max( 1, min(round(nf_max/(4*2*d)), n_seeds) );
    expini_iter = 0;
    while (size(Vbound,1) < d+1) && (expini_iter < expini_max)
        expini_iter = expini_iter+1;
        Vavoid = [k0v; cv_temp];
        ik0 = choosekm(V0,Vavoid);
        % Note: no test here, km is chosen from seeds, which are assumed to be
        %       viable (not tested)
        k0=V0(ik0,:); % initial cartesion expansion point
        k0v = [k0v; k0];

        Vini =  expanellipini(funcion, k0,d, bmin,bmax);

        % reject boundary points overlapping w/ the other ellipsoids
        for iellip=1:size(Ev,3)
            [~, IVini] = ellipdiff(Vini, Ev(:,:,iellip), cv(:,:,iellip), d);
            Vbound_out = [Vbound_out; Vini(~IVini,:)];
            Vini = Vini(IVini,:);
        end
        Vbound = [Vbound; Vini];
    end

    [~, nfeval_mid] = feval(funcion,'-g');
    nfeval_expini = nfeval_mid - nfeval_pre;
    if debugging
        fprintf('[DEBUG] expanellip_wrapper, fevals used in the initial expansion: %d/%d.\n', nfeval_expini, nf_max);
    end

    %%%% Shell sampling incl. initial expansion points
    % adjust number of total samples for the expansions
    nexp_max = max(0, nf_max - nfeval_expini);

    % expand over viable subspace not overlapping w/ the other ellipsoids
    Vexp = [V0; Vbound];
    [V,~,E,c,v] = expanellip(funcion, Vexp,d, nexp_max,nexpmin,...
                      bmin,bmax, Ev,cv, grate, tol);

    [~, nfeval_post] = feval(funcion,'-g');
    nfeval_exp = nfeval_post - nfeval_pre;
    nfeval = nfeval + nfeval_exp;

    if debugging
        vfreq = (size(V,1)-size(Vexp,1))/nfeval_exp;
        assert(vfreq <= 1);
        fprintf('[DEBUG] expanellip_wrapper, frequency of new points: %.3g.\n', vfreq);
    end

%     if debugging && (d == 2) && exist('ellipse', 'file')
%         scatterellipse(V0,...
%             [V; k0v; Vbound],...
%             [ones(size(V,1),1); 1.5*ones(size(k0v,1),1); 2*ones(size(Vbound,1),1)],...
%             cat(3,Ev,E), cat(3,cv,c),...
%             '[DEBUG] expanellip wrapper');
%     end

end





function [V_ca, nf_max_vec, nc, Ev,cv,vv] = expanellip_prep_seeds(Vtotal, IDX, Ev0,cv0,vv0, nf_max)
    assert(max(IDX) == numel(vv0));

    % re-arrange cover based on number of samples per ellipsoid
    % prop to volumes, sort by volume (=sample size)
    [nf_max_vec, I] = sort(splitlrm(nf_max, vv0(:)), 1, 'descend');

    % skip the small volume cover elements
    nc0 = numel(nf_max_vec); I0 = I;
    nc = find(nf_max_vec,1,'last');
    nf_max_vec = nf_max_vec(1:nc); I = I(1:nc);
    Ev = Ev0(:,:,I); cv = cv0(:,:,I); vv = vv0(I);

    % re-assign seeds of the small volume cover elements to the large ones
    for j=(nc+1):nc0
        IVtot = (IDX==I0(j));
        IDX(IVtot) = dsearchn(ellipcentrows(cv), Vtotal(IVtot,:));
    end

    % gather seeds
    % group as clustered; no special treatment of ellipsoids overlaps
    V_ca=cell(nc,1);
    for j=1:nc
        V_ca{j}=Vtotal(IDX==I(j),:);
    end
end



function cv_row = ellipcentrows(cv)
%ELLIPCENTROWS Convert matrix of d-dimensional ellipsoids centers into
% a row-vectors matrix.
%
% Syntax:
%     cv_row = ellipcentrows(cv)
%
% Inputs:
%     cv  = d x 1 x k matrix with centers of k ellipsoids.
%
% Outputs:
%     cv_row  = k x d matrix with centers of k ellipsoids.
%
    if size(cv, 3) == 1, sd = 1; else sd = 2; end
    cv_row = shiftdim(cv, sd);
end
