function [cost] = shellsTangent(x, r_small, r_large, tp_dim)
%SHELLSTANGENT Distance to a closer of the two shells at (r_small+r_large)/2:
% first around the origin, and a second tangent to the first one on the right
% in the tp_dim (by default: 1).
    if nargin < 4
        tp_dim = 1;
    end
    c2 = zeros(1,numel(x)); % = c1
    c2(tp_dim) = 2*r_large;
    cost= min(hstest.shell(x, r_small, r_large), hstest.shell(x-c2, r_small, r_large));
end
