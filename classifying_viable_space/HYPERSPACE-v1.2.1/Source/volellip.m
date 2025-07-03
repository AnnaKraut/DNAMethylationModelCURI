%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%volellip.m
%
%Calculates the volume of a single multidimensional ellipsoid
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Vol=volellip(dim,a)
%
%Input:
%       dim = dimension of the ellipsoid
%       a = vector with the lengths of ellipsoid axes
%
%Output:
%       Vol = volume of a multidimensional ellipsoid
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function auxs=volellip(dim,a)
assert(isvector(a));
% @MR, TODO rm `dim` param from here and from calls
assert(numel(a)==dim);

auxs=prod(a)*volsph(dim);

end
