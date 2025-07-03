function [x] = rowfevaln(func, nout, B)
%ROWFEVALN Evaluate func at each row of B collecting each of the first
% nout outputs.
%
    n = size(B,1);
    x = nan(n,nout);
    xiCell = cell(1,nout);
    for i=1:n
       [xiCell{1:nout}]=feval(func,B(i,:));
       x(i,:) = cell2mat(xiCell);
    end
end
