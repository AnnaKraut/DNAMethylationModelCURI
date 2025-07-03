function [v] = volsph(n)
%VOLSPH Compute volume of a unit n-sphere.
%
% Closed form variant as compared to the recursive `volsphrec`.

% Accuracy comparison over dimensions:
%     for i=0:100
%         v1=volsphrec(i); v2=volsph(i);
%         relerr=abs(v1 - v2)/v1;
%         fprintf('\tdim=%3d, relerr=%.3g\n',i,relerr);
%         if relerr > 1e-12, warning('Inconsistency for dim=%d.',i); end
%     end
% Accurate up to 1e-14 -- 1e-16 relative error.
%
% Timing comparison:
%     d=30; n=1e4;
%     tic; for i=1:n, volsphrec(d); end; toc
%     tic; for i=1:n, volsph(30); end; toc
% Matlab R2014a (repeatable):
%     Elapsed time is 1.146056 seconds.
%     Elapsed time is 0.190377 seconds.
%

    if n==0,  v=1; return; end

    vln = log(2)+(n/2)*log(pi)-log(n)-gammaln(n/2);
    v = exp(vln);

end
