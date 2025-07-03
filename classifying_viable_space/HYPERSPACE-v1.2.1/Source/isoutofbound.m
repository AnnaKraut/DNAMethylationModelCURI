function [bool] = isoutofbound(y, bmin, bmax)
%ISWITHINBOUND Row-wise check if point `y` is out of `bmin` or `bmax`
% bounds.
%
    bool = any(bsxfun(@gt, y, bmax) | bsxfun(@lt, y, bmin) ,2);
%     if isdebugging
%         fprintf('[DEBUG] isoutofbound: %d / %d.\n',sum(bool),numel(bool));
%     end
end
