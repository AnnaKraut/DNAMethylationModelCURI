function [cost] = twoBalls(v, varargin)
    if nargin > 1
        x = varargin{1};
    else
        x = 1;
    end
    c = x*ones(1,numel(v));
    v1 = v + c;
    v2 = v - c;
    cost=min(norm(v1), norm(v2));
end
