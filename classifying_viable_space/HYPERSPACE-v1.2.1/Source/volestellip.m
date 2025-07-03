function [ vol, res, err, nint, E, c, v, eidx ] = volestellip(V,dim, n, bmin,bmax, k)
%VOLESTELLIP Estimate volume of space occupied by points in rows of V,
% using n uniform samples drawn from minimal volume enclosing ellipsoids
% (subject to bmin and bmax bounds).
%
% Take a median of successful from k computations of the ellipsoids cover.
%
debugging = isdebugging;

    assert(~any(isoutofbound(V,bmin,bmax)));

    % TODO implement adaptive sample size
    % if not given, repeat as many times as the max nr of clusters
    if nargin < 6
        k = nclusmax(n, dim);
    end

    vol_vec=inf(1,k);
    % Take minimal volume (integral of a unit function) out of nRepeat
    % clusterizations of points into ellipsoids and uniform sampling from
    % them
    for i = k:-1:1 % from the top for implicit cell arrays allocation
        % compute the cover
        try
            [Ei, ci, vi, eidxi] = coverellipsoid(V,dim);
        catch ME
            if strcmp(ME.identifier, 'COVERELLIPSOID:SingularMVEE')
                warning('VOLESTELLIP:SingularMVEE', 'Failed to compute the ellipsoids cover.');
                continue;
            else
                rethrow(ME);
            end
        end
        % compute volume of the cover
        % intestellip: randellipsoid => integrator (eval unidad only if not isoutofbound)
        [resi, erri, ninti] = intestellip('unidad',1, Ei,ci,vi,dim, n, bmin,bmax);
        vol_vec(i) = sum(resi);
        if sum(ninti) == 0 % check also nclus condition: all(nint0 >= 2*(dim+1)) ?
            warning('VOLESTELLIP:OutOfBounds', 'All points sampled from the ellipsoids cover are out of bounds.');
            assert(vol_vec(i)==0);
        end


        res_ca{i} = resi;
        err_ca{i} = erri;
        nint_ca{i} = ninti;
        E_ca{i} = Ei;
        c_ca{i} = ci;
        v_ca{i} = vi;
        eidx_ca{i} = eidxi;
    end

    if all(isinf(vol_vec))
        error('VOLESTELLIP:SingularCover', 'Couldn''t estimate volume; the ellipsoid cover always contains singular MVEE.');
    end

    Isuccess = ~isinf(vol_vec); % Note: not isfinite(), to avoid errorneous NaN slipping over here

    % find cover element used as an estimate
    Iok = Isuccess;
    volok_vec = vol_vec(Iok);
    volMean = mean(volok_vec);
    volStd = std(volok_vec);
    if all(volok_vec == 0) % pick the one w/ the smallest volume of the cover
        volellipok_vec = cellfun(@sum, v_ca(Iok));
        [~, Jm] = min(volellipok_vec);
    else % pick the positive one closest to the mean
        Iok = Iok & (vol_vec > 0);
        volok_vec = vol_vec(Iok);
        [~, Jm] = min(abs(volok_vec - volMean)); % note: mean w/ zeros
    end
    Im = find(Iok, Jm); Im = Im(end);

    vol = vol_vec(Im);
    res = res_ca{Im};
    err = err_ca{Im};
    nint = nint_ca{Im};
    E = E_ca{Im};
    c = c_ca{Im};
    v = v_ca{Im};
    eidx = eidx_ca{Im};

    if debugging
        assert(sum(res) == vol);
        fprintf('[DEBUG] volestellip, est. over %d/%d successful repeats, volume (%d ellipsoids) = %.3g +/- %.3g (mean= %.3g, std = %.3g).\n',...
            sum(Isuccess), k,...
            numel(res), sum(res), sum(err),...
            volMean, volStd);
        %fprintf('[DEBUG] volestellip, est. over %d repeats, min. volume (%d ellipsoids) = %.3g +/- %.3g.\n', nRepeat, numel(res), sum(res), sum(err));
    end

end
