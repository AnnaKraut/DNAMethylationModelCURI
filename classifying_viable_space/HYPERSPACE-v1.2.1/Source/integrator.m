%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%integrator.m
%
%Performs a Monte Carlo integration of a function inside an ellipsoid.
%The sampled points that lie inside a set of other ellipsoids are
%not counted
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       [res, err, nib]=integrator(funcion,nfout,B,Vol,bmax,bmin,EjesA,centccc,dim)
%
%Input:
%       funcion = function to integrate; each output is integrated
%       nfout = number of outputs of the function (integrated separately)
%       separately and has to be of the same type (usually double)
%       B = set of points to be sampled inside the ellipsoid.
%       Vol = volume of the ellipsoid.
%       bmax = vector with the upper bound of our parameter space.
%       bmin = vector with the lower bound of our parameter space.
%       EjesA = This matrix (dimxdimx(n ellipsoids) contains all the
%             information regarding the set of other ellipsoid where we
%             will test if the points of B lie there.
%       centccc = center of the set of other ellipsoids.
%       dim = dimension of the parameter space
%
%Output:
%        res = vector of lenght nfout, with results of the MC integration
%              within the ellipsoid and the [bmin, bmax] box
%        err = vector of errors of this integration.
%        nib = number of points outside of EjesA and within bounds
%              (nib <= size(B,1))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [res, err, nib]=integrator(funcion,nfout,B0,Vol,bmax,bmin,EjesA,centccc,dim)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % n: number of points for MC integration
    n=size(B0,1);
   
    % nelip: number of other ellipsoids
    if isempty(EjesA)
        nelip=0;
    else
        S=size(EjesA);
        if numel(S)==2
            nelip=1;
        else
            nelip=S(3);
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Points filtering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % filter out points that are out of bounds
    B = B0(~isoutofbound(B0,bmin,bmax),:);

    % filter out points that are in other ellipsoids
    for j=1:nelip
        B=ellipdiff(B,EjesA(:,:,j),centccc(:,:,j),dim);
    end
    nib = size(B,1);

    if (nib == 0)
        %no volume
        res=zeros(nfout,1);
        err=zeros(nfout,1);
        return
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MC integration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    X = rowfeval(funcion, nfout, B);
    % Estimate & error of the Monte Carlo integration
    % Note: assuming Vol was estimated using all n points, hence, using n,
    %       not nib, for integral estimation
    %
    [res, err] = intmcest(Vol, X, n);
    res = res';
    err = err';
end
