function [ vol , X_unif] = computeMCVolume_LineSegmentsAndBalls( LineSegments , Balls , n_samples )
% [vol] = COMPUTEMCVOLUME(f , LineSegments , Balls , n_samples) runs MC sampling over
% the provided hypercylinders / balls ensemble.
% This corresponds to the volume of the viable region that the cost
% function fDistToLineSegmentsAndBalls(..) define at threshold=0.
%
%Input:
%   LineSegments: is a cell array with structs of fields:
%      .a , .b : start / end
%      .r      : radius (if not specified zero)
%
%   Balls:        is a cell array with structs of fields:
%      .c       : center
%      .r       : radius
%
%
% Thomas Liphardt, June 2016
%

% determine dimension:
try
    dim = size(LineSegments{1}.a,1);
end
try
    dim = size(Balls{1}.c,1);
end


% computes the volume of the n-dimensional ball
f_vol_ball = @(r1,dim1) (( pi^(dim1/2) ) / gamma( (dim1/2 )+ 1)) * r1^dim1;

% computes the volume of the volume defined by:
% dist(x)<r , x \in "line from a to b"
% this is: (volume of ball with radius r) + (volume of hypercylinder with axis from a to be with radius r)
f_vol_linesegment = @(r1,a1,b1,dim1) f_vol_ball(r1,dim1) + norm(b1-a1)*f_vol_ball(r1,dim1-1);

try, dim = size(LineSegments{1}.a,1); end
try, dim = size(Balls{1}.a,1); end

% 1. assign number of samples to different bodies:
l_volumes = [];
b_volumes = [];
l_nsamp   = [];
b_nsamp   = [];
for zi=1:numel(LineSegments),
    ls = LineSegments{zi};
    l_volumes(zi) = f_vol_linesegment(ls.r,ls.a,ls.b,dim);
end
for zi=1:numel(Balls),
    lb = Balls{zi};
    b_volumes(zi) = f_vol_ball(lb.r,dim);
end

all_volumes = sum(l_volumes)+sum(b_volumes);

% number of samples:
for zi=1:numel(LineSegments), l_nsamp(zi) = ceil( n_samples * (l_volumes(zi)/all_volumes) ) ; end
for zi=1:numel(Balls), b_nsamp(zi) = ceil( n_samples * (b_volumes(zi)/all_volumes) ) ; end


% 2.1 Generate unif. samples over line segment distance volumes:
L_Xunif = {};
for zi=1:numel(LineSegments),
    Li = LineSegments{zi};
    L_Xunif{zi} = unif_linesegment(Li.a,Li.b,Li.r,dim,l_nsamp(zi));
end

% 2.2 Generate unif. samples over each ball:
B_Xunif = {};
for zi=1:numel(Balls),
    Bi = Balls{zi};
    B_Xunif{zi} = unif_ball(Bi.c,Bi.r,dim,b_nsamp(zi));
end



fprintf('\n\nWelcome to MCVolume!\n\n');

% 3. Handle overlaps, i.e. process first line segments in natural order and
%    then balls in natural order, and we
%    compute for each the overlap with all ealier bodies.
%    In case of overlap, we remove the overlapping samples and adjust the
%    volume (->nonoverlap_volumes).
l_nonoverlap_volumes = [];
b_nonoverlap_volumes = [];

flag_PlotSamples = 1;

if(flag_PlotSamples && (dim==2 || dim==3) ) , figure(); hold on; end

for zi=1:numel(LineSegments),
    % overlap with all those will be removed
    predecessors = 1:zi-1;

    % here we store all indeces of overlap samples
    x_overlapping = zeros(1,size(L_Xunif{zi},2))~=0;
    for pe=predecessors;
        for zj=1:size(L_Xunif{zi},2),
            if( hstest.wurst.distancePoint2Line( LineSegments{pe}.a , LineSegments{pe}.b , L_Xunif{zi}(:,zj) , 'segment' ) < LineSegments{pe}.r ),
                x_overlapping(zj) = 1;
            end
        end
    end

    % remove all overlapping and adjust volume..
    l_nonoverlap_volumes(zi) = (1-( sum(x_overlapping) / l_nsamp(zi) )) * l_volumes(zi);

    if( flag_PlotSamples ),
        col_rand = rand(1,3);
        if(dim==2), line(L_Xunif{zi}(1,:),L_Xunif{zi}(2,:),'Marker','.','LineStyle','none','Color',col_rand); end
        if(dim==3), line(L_Xunif{zi}(1,:),L_Xunif{zi}(2,:),L_Xunif{zi}(3,:),'Marker','.','LineStyle','none','Color',col_rand); end
    end

    L_Xunif{zi}(:,x_overlapping~=0) = [];

    if( flag_PlotSamples ),
        if(dim==2), line(L_Xunif{zi}(1,:),L_Xunif{zi}(2,:),'Marker','x','LineStyle','none','Color',col_rand); end
        if(dim==3), line(L_Xunif{zi}(1,:),L_Xunif{zi}(2,:),L_Xunif{zi}(3,:),'Marker','x','LineStyle','none','Color',col_rand); end
    end
end

for zi=1:numel(Balls),
    % overlap with all those will be removed
    predecessors_l = 1:numel(LineSegments);
    predecessors_b = 1:zi-1;

    % here we store all indeces of overlap samples
    x_overlapping = zeros(1,size(L_Xunif{zi},2))~=0;

    % check overlap with lines (all lines)
    for pe=predecessors_l;
        for zj=1:size(B_Xunif{zi},2),
            if( hstest.wurst.distancePoint2Line( LineSegments{pe}.a , LineSegments{pe}.b , B_Xunif{zi}(:,zj) , 'segment' ) < LineSegments{pe}.r ),
                x_overlapping(zj) = 1;
            end
        end
    end

    % check overlap with predecessor balls
    for pe=predecessors_b;
        for zj=1:size(B_Xunif{zi},2),
            if( norm( B_Xunif{zi}(:,zj) - Balls{pe}.c ) < Balls{pe}.r ),
                x_overlapping(zj) = 1;
            end
        end
    end

    % remove all overlapping and adjust volume..
    b_nonoverlap_volumes(zi) = (1-( sum(x_overlapping) / b_nsamp(zi) )) * b_volumes(zi);

    if( flag_PlotSamples ), col_rand = rand(1,3);
        if(dim==2),line(B_Xunif{zi}(1,:),B_Xunif{zi}(2,:),'Marker','.','LineStyle','none','Color',col_rand); end
        if(dim==3),line(B_Xunif{zi}(1,:),B_Xunif{zi}(2,:),B_Xunif{zi}(3,:),'Marker','.','LineStyle','none','Color',col_rand); end
    end

    B_Xunif{zi}(:,x_overlapping~=0) = [];

    if( flag_PlotSamples ),
        if(dim==2), line(B_Xunif{zi}(1,:),B_Xunif{zi}(2,:),'Marker','o','LineStyle','none','Color',col_rand); end
        if(dim==3), line(B_Xunif{zi}(1,:),B_Xunif{zi}(2,:),B_Xunif{zi}(3,:),'Marker','o','LineStyle','none','Color',col_rand); end
    end
end


% figure(); hold on; for zi=1:numel(B_Xunif), plot(B_Xunif{zi}(1,:),B_Xunif{zi}(2,:),'.b'); end; for zi=1:numel(L_Xunif), plot(L_Xunif{zi}(1,:),L_Xunif{zi}(2,:),'.r'); end   %%%%%%%

vol = sum(l_nonoverlap_volumes)+sum(b_nonoverlap_volumes);

fprintf('\n\nEnsemble statistics:\n');
fprintf('%d LineSegments , %d Balls \n',numel(LineSegments),numel(Balls));
fprintf('Ensemble volume = %8.4f\n',vol );
fprintf('Sum of single volumes  = %8.4f\n',sum(l_volumes)+sum(b_volumes));
%fprintf('Overlap Ratio             = %5.3f%%', sum(volumes)-sum(overlap_volumes));


fprintf('\nVol=%8.4f\nDone!\n\n',vol);


X_unif = [L_Xunif{:},B_Xunif{:}];


end


% tests for all X (with column vectors) whether they are within the
% ellpsoid   (c-x)'*E*(c-x) < 1
function [in_ellipse] = test_ellip(E,c,X),
    X = bsxfun(@minus,c,X);
    in_ellipse = dot(X,(E*X))<=1;
end

function  Xb = unif_ball(c,radius,dim,n),
    Xb = randn(dim,n);
    Xb = bsxfun( @rdivide , Xb , sqrt(sum(Xb.^2,1) ));
    Xb = bsxfun( @times   , Xb , rand(1,n).^(1/dim) );
    Xb = Xb * radius;
    Xb = bsxfun( @plus , Xb , c );
end

function Xl = unif_linesegment(a,b,radius,dim,n),
    % 1. distribute samples into:
    %    a) "end"-ball-A
    %    b) "end"-ball-B
    %    c) cylinder


    % computes the volume of the n-dimensional ball
    f_vol_ball = @(r,dim) ( ( pi^(dim/2) ) / gamma( (dim/2 )+ 1) ) * r^dim;

    % computes the volume of the volume defined by:
    % dist(x)<r , x \in "line from a to b"
    % this is: (volume of ball with radius r) + (volume of hypercylinder with axis from a to be with radius r)
    f_vol_linesegment = @(r,a,b,dim) f_vol_ball(r,dim) + norm(b-a)*f_vol_ball(r,dim-1);


    vol_eb  = f_vol_ball(radius,dim);
    vol_cyl = norm(b-a)*f_vol_ball(radius,dim-1);

    vol_tot = vol_eb+vol_cyl;

    s_ab  = ceil(((vol_eb)  / vol_tot) * n);
    s_cyl = ceil(((vol_cyl) / vol_tot) * n);

    % sample ball:
    Xab = unif_ball(zeros(dim,1),radius,dim,s_ab);

    % distribute into two end-balls:
    side = (b-a)'*Xab;

    Xa   = Xab(:,side<0); Xb = Xab(:,side>0);
    Xa   = bsxfun(@plus,Xa,a);
    Xb   = bsxfun(@plus,Xb,b);

    % figure(); hold on; plot3(Xa(1,:),Xa(2,:),Xa(3,:),'.r'); plot3(Xb(1,:),Xb(2,:),Xb(3,:),'.b');

    % sample cylinder:
    % we sample the cylinder where the axis goes from [0,0,..,0] to [ norm(b-a) ,0,..,0], then
    % we transform those samples into the correct position..
    X_cyl = [ rand(1,s_cyl)*norm(b-a) ; unif_ball(zeros(dim-1,1), radius ,  dim-1,s_cyl)];

    % now transform, such that:
    % L( [0,0,..,0] -> [1,0,..,0]) becomes L( O -> (b-a) ):
    % We create two orthogonal bases, and then compute the transform:
    Basis_A = [ [1,zeros(1,dim-1)]' , null(  [1,zeros(1,dim-1)] )];
    bb_normed = (b-a) / norm((b-a));
    Basis_B = [  bb_normed , null(  bb_normed') ];
    T_rot   = Basis_A \ Basis_B;

    % now we can do the full transform:
    X_cyl = bsxfun( @plus , a , (T_rot * X_cyl) ) ;

    % figure(); hold on; plot(Xa(1,:),Xa(2,:),'.r'); plot(Xb(1,:),Xb(2,:),'.b'); plot(X_cyl(1,:),X_cyl(2,:),'.k') ;
    % figure(); hold on; plot3(Xa(1,:),Xa(2,:),Xa(3,:),'.r'); plot3(Xb(1,:),Xb(2,:),Xb(3,:),'.b'); plot3(X_cyl(1,:),X_cyl(2,:),X_cyl(3,:),'.k') ;

    Xl = [Xa,Xb,X_cyl];
end
