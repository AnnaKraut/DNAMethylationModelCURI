%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%volsphrec.m
%
%Calculate the volume of a multidimensional sphere with unit radius
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Vol = volsphrec(dim)
%
%Input: 
%       dim = dimension of the parameter space
%       
%Output: 
%       Vol = volume of a multidimensional sphere with unit radius
%             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function auxs=volsphrec(dim)


if (dim==0)
    
    auxs=1;
    
elseif (dim==1)
    
    auxs=2;
    
else
    
    auxs=((2*pi)/(dim))*volsphrec(dim-2);
    
end
