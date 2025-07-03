function [nv] = splitlrm(n, v)
%SPLITLRM Split integer `n` according to proportions `v` using the largest
% reminder method (LRM).
%
% Note: n is rounded down to an integer.
%
    n = floor(n);
    if(any(v<0) || ~any(v>0))
        error('SPLITLRM:IllegalArgument','Negative proportion or not a positive sum of the proportions.');
    end
    nvprop = n * v / sum(v);
    nv = floor(nvprop);
    nvrem = nvprop - nv;
    nrem = n - sum(nv);
    if nrem > 0
        [~, nvremtopidx] = sort(nvrem, 'descend');
        nvremtopidx = nvremtopidx(1:nrem);
        nv(nvremtopidx) = nv(nvremtopidx) + 1;
    end
    %assert(sum(nv) == n);
end
