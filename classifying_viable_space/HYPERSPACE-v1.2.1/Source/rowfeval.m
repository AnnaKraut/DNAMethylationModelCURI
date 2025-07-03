function [ x ] = rowfeval( func, nout, B )
%ROWFEVAL Evaluate func at each row of B collecting each of the first
% nout outputs.
%
    if nout == 1
        % significantly faster w/o cell shenenigans
        x = rowfeval1(func, B);
    else
        x = rowfevaln(func, nout, B);
    end
end
