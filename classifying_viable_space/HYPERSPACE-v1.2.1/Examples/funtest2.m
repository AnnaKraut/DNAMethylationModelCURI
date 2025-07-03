function [cost] = funtest2(v, varargin)
    if nargin > 1
        x = varargin{1};
    else
        x = 1;
    end
    c = x*ones(1,numel(v));
    v1 = v + c;
    v2 = v - c;
    cost=min(funtest(v1), funtest(v2));
end
