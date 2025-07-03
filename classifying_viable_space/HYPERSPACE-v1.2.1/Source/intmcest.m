function [ res, err ] = intmcest( vol, X, n )
%INTMCEST Naive Monte Carlo integral estimate of uniform samples (rows of X)
% from a domain with volume vol.
%
% Note: uses explicit samples number n, which may be bigger than number of
%       rows of X, which is equivalent to treating filtered out samples as 0.
%

% result: uniform sample mean = volume * est. of mean
meanEst = sum(X)/n;
res=vol*meanEst;
% error: std. dev. of sample mean (std. error) = volume * sqrt(est. of var / n)
% Note: using unbiased variance estimator, i.e.
%    [1] (1/(n-1)) * sum_i((X_i-meanEst)^2) =
%    [2] = (n/(n-1)) * (sum_i(X_i.^2/n) - meanEst^2)
% where X_i denotes i-th sample (row of X).
if n > 1
    % [1] w/ bsxfun is actually slightly faster (R2014a) than [2]
    varEst = sum(bsxfun(@minus, X, meanEst).^2)/(n-1);
else
    varEst = 0;
end
err=vol*sqrt(varEst/n);

end
