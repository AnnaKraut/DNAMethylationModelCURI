function Out = Volint(funcion,threshold, V, bmax,bmin, n, varargin)
%VOLINT Monte Carlo estimation of the viable volume (and the cost integral)
% using minimal volume encapsulating ellipsoids cover.
%
% Returns a set of uniformly distributed viable points and the cover.
%
%
% Syntax:
%
%       Out = Volint(function,threshold, V, bmax,bmin, n)
%       Out = Volint(___, OPTIONS)
%
%
% Inputs:
%     function : function that recieves a parameter point and checks its cost.
%     threshold : scalar that defines the maximum value of the viable cost
%
%     V : matrix with viable parameter points. Every vaible parameter
%         point corresponds to a row of the matrix
%
%     bmax : row  vector with the upper bound of our parameter space
%     bmin : row vector with the lower bound of our parameter space
%
%     n : scalar that defines the number of parameter
%         evaluations carried out in the Monte Carlo integration
%
%     OPTIONS : structure or name-value pairs with internal parameters of
%               the algorithm, or extra parameters for the cost function.
%          OPTIONS.nint  : number of ellipsoid cover integration points;
%                          sampled (not evaluated) to find the initial
%                          volume estimate.
%          OPTIONS.nrandmax  : number of ellipsoid cover integration points;
%          OPTIONS.fargs : cell with extra parameters for the cost
%                          function
%
% Outputs:
%     Out : structure with the results
%
%         Out.V     : matrix with the set of uniformly distributed viable
%                     parameter points. Every vaible parameter point
%                     corresponds  to a row of the matrix
%         Out.cost  : column vector that contains the cost of the viable
%                     parameter points present in Out.V
%
%         Out.nfeval: number of function evaluations
%
%         Out.vol   : scalar with the  estimation of the viable volume.
%         Out.err   : scalar with estimation of the error in the viable
%                     volume.
%
%         Out.ci    : scalar with the estimation of cost integral over the
%                     viable volume.
%         Out.cierr : scalar with estimation of the error in the cost
%                     integral.
%
%         Out.ellip : strcuture describing ellipsoids used to estimate the
%                     viable volume and the cost integral
%             Out.ellip.A : (dim)x(dim)x(n elip) axes in cols
%             Out.ellip.c : (dim)x(1)x(n elip) centers in cols
%             Out.ellip.v : (n elip) volumes
%

dim=size(V,2);
if dim~=size(bmax,2) || dim~=size(bmin,2)
    error('VOLINT:IncompatibleShapes',...
        'V, bmax, and bmin muss have the same column size (points dimension).');
end

if size(V,1) < 10*(dim+1)
    error('VOLINT:NotEnoughPoints',...
        ['Not enough viable parameter points given.'...
        'The minimum required number of rows in V is 10*(dim+1)=%d.'],...
        10*(dim+1));
end

OPTIONS = parseVolintOptions(dim, varargin{:});
nint = OPTIONS.nint;
nrandmax = OPTIONS.nrandmax;
fargs = OPTIONS.fargs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filter by bounds and clean
Vtemp=V(:,1:dim);
Vtotal=Vtemp(~isoutofbound(Vtemp,bmin,bmax),:);
if isdebugging
    fprintf('[DEBUG] Volint, Number of points init = %d, and after bounds check = %d.\n',size(V,1),size(Vtotal,1));
end
%Vtotal=cleaning(Vtotal,dim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Choosing the integration domain %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[sumat, res0,err0,~, EjesA,centccc,Volumenelip] = volestellip(Vtotal,dim, nint, bmin,bmax);
if isdebugging
    fprintf('[DEBUG] Volint, Estimated init volume (%d ellipsoids, w/o threshold filtering) = %.3g +/- %.3g.\n',numel(res0),sumat,sum(err0));
end

% number of points to sample, taken into account the possible overshoot in
% volumes of the clustering-based ellipsoids
nintf = min(floor(n*sum(Volumenelip)/sumat), nrandmax);
% Note: when sumat = 0, we have min(Inf, nrandmax)


% prepare function to evaluate
functionfinal2('-s', threshold, nintf, dim, funcion, fargs);

% intestellip: randellipsoid => integrator but now eval given cost funcion,
%             accounting for both bounds and the threshold (cf. threshold and
%             viable global variables)
[res, err]=intestellip(@functionfinal2,2, EjesA,centccc,Volumenelip,dim, nintf, bmin,bmax);
[viable, nfeval] = functionfinal2('-g');

% viable volume integral
Out.vol=sum(res(1,:));
Out.err=sum(err(1,:));
if isdebugging
    fprintf('[DEBUG] Volint, Estimated final volume (%d ellipsoids, w/ threshold filtering) = %.3g +/- %.3g.\n',size(res,2),sum(Out.vol),sum(Out.err));
end

% viable points and their cost
nviable = nviablepts(viable);
Out.V=viable(1:nviable,1:dim);
Out.cost=viable(1:nviable,dim+1);
Out.nfeval=nfeval;
fprintf('[INFO] Volint, found %d viable samples in %d function evaluations (given max. %d).\n',nviable, nfeval, n);

% cost integral over viable volume
Out.ci=sum(res(2,:));
Out.cierr=sum(err(2,:));

% ellipsoids used for estimation
Out.ellip.A = EjesA;
Out.ellip.c = centccc;
Out.ellip.v = Volumenelip;

% MISC: 2-D ellipsoids diagnostics, cf. #8
if isdebugging && (dim == 2)
    scatterellipse(V, Out.V, Out.cost, Out.ellip.A, Out.ellip.c, '[DEBUG] Volint');
end

end



function optsout = parseVolintOptions(d, varargin)
    parser = inputParser;

    addParameter(parser, 'nint', 1e5, @isnonnegative);
    addParameter(parser, 'nrandmax', nrandmax(d), @isnonnegative);
    addParameter(parser, 'fargs', {}, @iscell);

    parse(parser, varargin{:});
    optsout = parser.Results;
end

function bool = isnonnegative(x)
    bool = isnumeric(x) && all(x(:) >= 0);
end
