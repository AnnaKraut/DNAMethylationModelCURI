function [ min_dist ] = fDistToLineSegmentsAndBalls( x , LineSegments , Balls )
%
%[dist] = FDISTTOLINESEGMENTSANDBALLS(LineSegments,Balls)
%
%computes the minimum distance to line segment / tubes / and balls
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

if(nargin<3), Balls = {}; end

min_dist = Inf;

for zi=1:numel(LineSegments),
    Li = LineSegments{zi};
    di = hstest.wurst.distancePoint2Line(Li.a,Li.b,x,'segment');
    if(isfield(Li,'r')),
        di = di - Li.r;
    end
    min_dist = min(min_dist,di);
end

for zi=1:numel(Balls),
    Bi = Balls{zi};
    di = sqrt(sum( (Bi.c-x).^2 ));
    if(isfield(Bi,'r')),
        di = di - Bi.r;
    end
    min_dist = min(min_dist,di);
end

% it is ok to return negative values..
% min_dist = max(min_dist,0);

end

