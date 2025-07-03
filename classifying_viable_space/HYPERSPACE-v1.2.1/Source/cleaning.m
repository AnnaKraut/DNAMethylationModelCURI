%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%cleaning.m
%
%Given a set of parameter points, it returns a smaller set necessary to define
%the minimum volume ellipsoid (MVE) of the initial set.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       [W IDI]=cleaning(V,dim)
%
%Input:
%       V = set of parameter points.
%       dim = dimension of the parameter space.
%
%Output:
%       W = subset of V necessary to define the MVE.
%       IDX = index for every point of W
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [auxs1 auxs2]=cleaning(Vfinalini,dim)
debugging = isdebugging();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



paso=2000;
W=zeros(0,size(Vfinalini,2));
W(1:paso,:)=randrows(Vfinalini,paso);
U=size(W,1);
j=1;
first=false;
Vfinal=Vfinalini;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Choose the neccesary points to build the MVE %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @MR, TODO explain loop invariant or add maxiter break
while (size(Vfinal,1)>1000)
    if debugging && (mod(j,100) == 1)
        fprintf('[DEBUG] cleaning, iterations #%d .. %d, started with nr of parameter points = %d.\n',j, j+100-1, size(Vfinal,1));
    end



    if (U>2*(dim+1)) && (~first)

        first=true;
        W(1:U,1:dim); % @MR, WTF?

    else

        V=randrows(Vfinal,paso);

        for marca=1:length(Volumenelip)

            V=ellipdiff(V,EjesA(:,:,marca),centccc(:,:,marca),dim);

        end


        lv=length(V(:,1));
        W(U+1:(U+lv),:)=V;

    end

    W=unique(W,'rows');
    U=size(W,1);

    [EjesA centccc Volumenelip]=coverellipsoid(W,dim);
    j=j+1;

    Vfinal=Vfinalini;

    for marca=1:length(Volumenelip)

        Vfinal=ellipdiff(Vfinal,EjesA(:,:,marca),centccc(:,:,marca),dim);

    end

end


%if the remaining points are less than 1000, they are added to W
V=Vfinal;
lv=length(V(:,1));
W(U+1:(U+lv),:)=V;
W=unique(W,'rows');

[nc IDX]=nclus(W(:,1:dim));


auxs1=W;
auxs2=IDX;
















