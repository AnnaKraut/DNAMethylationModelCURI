function [ x0 , f , threshold , bmin , bmax , volume , LineSegments , Balls ] = initRandWurst( dim )
%INITRANDWURST Create the test scenario of a dim-dimensional wurst
% (piece-wise cylinders and spheres at the joints).
%

    threshold = 0;
    n_LineSegments = 8;
    LineSegments_Length = 8;
    LineSegments_Radius = 2;
    BallsAt             = mod( 1:n_LineSegments , 2 ) == 0;
    BallsRadius         = 3.0;

    LineSegments = {};
    Balls        = {};


    % specifies the widths of the bounding box relative to the min required
    % widths.
    BoundingBoxBadnessRatio = 4.0;

    AllPoints = zeros(dim,0);

    LineStart    = zeros(dim,1);
    AllPoints    = [ AllPoints , LineStart ];

    x0 = LineStart;

    for zi=1:n_LineSegments,

        Move_i = randn(dim,1);
        Line_i = LineStart + LineSegments_Length * 0.5 * ( (ones(dim,1)/norm(ones(dim,1))) + (Move_i/norm(Move_i)) ) ;

        LineNext = Line_i;
        LineSegments{zi} = struct('a',LineStart,'b',LineNext,'r',LineSegments_Radius);
        AllPoints = [ AllPoints , LineNext ];

        if(BallsAt(zi)),
            Balls{end+1} = struct('c',LineStart,'r',BallsRadius);
        end
        LineStart = LineNext;
    end

    % create the cost function
    costfun = @(x) hstest.wurst.fDistToLineSegmentsAndBalls(x(:),LineSegments,Balls);

    % compute volume of this ensemble, assuming that the threshold is zero:
    [ vol_ensemble , X_ensemble ] = hstest.wurst.computeMCVolume_LineSegmentsAndBalls( LineSegments , Balls , 10000);

    % create plot of the function values:
    if(dim==2),
        X_ensemble_cost = nan(1,size(X_ensemble,2));
        for zi=1:numel(X_ensemble_cost), X_ensemble_cost(zi) = costfun(X_ensemble(:,zi)); end,
        figure(); scatter(X_ensemble(1,:),X_ensemble(2,:),50,X_ensemble_cost,'.');
    end

    if(dim==3),
        X_ensemble_cost = nan(1,size(X_ensemble,2));
        for zi=1:numel(X_ensemble_cost), X_ensemble_cost(zi) = costfun(X_ensemble(:,zi)); end,
        figure(); scatter3(X_ensemble(1,:),X_ensemble(2,:),X_ensemble(3,:),50,X_ensemble_cost,'.');
    end


    %figure(); plot(AllPoints(1,:),AllPoints(2,:));
    %figure(); plot3(AllPoints(1,:),AllPoints(2,:),AllPoints(3,:));

    % compute bounding box:
    bmax = -inf(1,dim); bmin = inf(1,dim);
    for zi=1:dim,
        max_i = max(AllPoints(zi,:));
        min_i = min(AllPoints(zi,:));
        center_i = (max_i+min_i)/2;
        minwidth_i = max_i-min_i;
        bmax(zi)  = center_i + 0.5*(BoundingBoxBadnessRatio*minwidth_i);
        bmin(zi)  = center_i - 0.5*(BoundingBoxBadnessRatio*minwidth_i);
    end

    x0 = transpose(x0(:)); % make sure output is a row vector
    volume = vol_ensemble;
    f = costfun;
end

