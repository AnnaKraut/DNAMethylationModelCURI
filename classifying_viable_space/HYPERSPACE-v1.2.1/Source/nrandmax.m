function nmax = nrandmax(d)
%NRANDMAX Maximal number of random samples, each being d-dimensional.
%
% Examples:
%    % randn calls with nrandmax(d) all under 2.9 sec on MATLAB R2014a, 8GB RAM, Intel(R) Core(TM) i7-3615QM CPU @ 2.30GHz, 256 KB Cache
%    for d=41:-5:1; tic; randn(nrandmax(d), d); et(d) = toc; fprintf('%2d in %.3g sec\n',d,et(d)); end
%    % randsphere calls with nrandmax(d) all under 11 sec in the same setup as above (max. for d=7,...,11)
%    for d=41:-5:1; tic; randsphere(nrandmax(d), d, 1); et(d) = toc; fprintf('%2d in %.3g sec\n',d,et(d)); end
%

    %
    [~,maxsize] = computer;
    nmax = round(maxsize / (2e6 * max(d,8)));

end
