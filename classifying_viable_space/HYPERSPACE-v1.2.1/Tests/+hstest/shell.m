function [cost] = shell(x, r_small, r_large)
%SHELL Distance to a shell at (r_small+r_large)/2, around the origin.
    cost=abs(norm(x)-((r_small+r_large)/2));
end
