%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%testkm.m
%
%Checks the viability of a given parameter point.
%If it is not viable it returns a new parameter point
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       index=testkm(funcion,V,km,bmax,bmin)
%
%
%Input:
%       funcion = function that returns 1(0)
%                 if the parameter point V(km,:) is viable (nonviable).
%       V = set of parameter points
%       km = index of the parameter point V(km,:).
%       bmax = upper bound of the parameter space.
%       bmin = lower bound of the parameter space.
%
%Output:
%       index= index of a viable parameter point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function auxs=testkm(funcion,V,km,bmax,bmin)

resu=1;
S=size(V,1);


A=feval(funcion,V(km,:)); %check if the parameter is viable

if (A==0)

    resu=0;

elseif isoutofbound(V(km,:),bmin,bmax)

    resu=0;

end

if resu==1

    auxs=km;

else
    rng('shuffle');
    num=(S-1)*rand+1;
    sal=testkm(funcion,V,floor(num),bmax,bmin);
    auxs=floor(sal);

end

end
