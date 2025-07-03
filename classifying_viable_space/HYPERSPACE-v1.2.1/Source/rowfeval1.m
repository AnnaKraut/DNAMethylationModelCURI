function [x] = rowfeval1(func, B)
%ROWFEVAL1 Evaluate func at each row of B collecting first output.
%

% Note: (for + feval + string fun handle) vs. (arrayfun + @ fun handle):
%     n = 1e5; m = rand(n, 3);
%     tic; x = arrayfun(@unidad, m); toc
%     tic; x = nan(1,n); for i=1:n, x(i) = feval('unidad', m(i,:)); end; toc
% in 2014a (repeatable):
%     Elapsed time is 1.601675 seconds.
%     Elapsed time is 0.817658 seconds.
% in 2015b (repeatable):
%     Elapsed time is 1.423428 seconds.
%     Elapsed time is 1.763228 seconds.

    n = size(B,1);
    x = nan(n, 1);
    for i=1:n
       x(i) = feval(func,B(i,:));
    end
end
