function Out = MCexp(fun,threshold,x0,bmax,bmin,n,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MCexp.m
%
%Obtains viable parameter points through the Out-of-equilibrium Adaptive
%Monte Carlo method
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Out = MCexp(function,threshold,x0,bmax,bmin,n)
%   __  = MCexp(__,OPTIONS)
%
%Input:
%       function : function that recieves a parameter point and checks its
%                  cost
%       threshold : scalar that defines the maximum value of the viable cost
%       x0 : row vector with the coordinates of a viable parameter point
%       bmax : row  vector with the upper bound of our parameter space
%       bmin : row vector with the lower bound of our parameter space
%       n : scalar that defines the maximum number of parameter evaluations
%
%       OPTIONS : structure or name-value pairs with internal parameters of the
%                 algorithm or extra parameters for the cost function.
%                OPTIONS.conv : boolean, if to check the convergence
%                               (default: false)
%                OPTIONS.tol : tolerance to accept the convergence
%                OPTIONS.nint : points sampled to check the convergence
%                OPTIONS.gr : number of parameter evaluations among updates
%                OPTIONS.Be : initial value of \Beta;
%                OPTIONS.viamax : maximum frequency of viable points that
%                                 maintains constant \Beta
%                OPTIONS.viamin : minimum frequency of viable points that
%                                 maintains constant \Beta
%                OPTIONS.tramax : maximum frequency of accepted transitions
%                                 that maintains constant the covariance matrix
%                OPTIONS.tramin : minimum frequency of accepted transitions
%                                 that maintains constant the covariance matrix
%                OPTIONS.tbemax : maximum relative change of \Beta among updates
%                OPTIONS.tsmax : maximum relative change of the covariance matrix
%                                among updates
%                OPTIONS.fargs : cell with extra parameters for the cost
%                                function
%
%
%       DEFAULT VALUES:
%       ===============
%                OPTIONS.tol=0.05;
%                OPTIONS.nint=1e5;
%                OPTIONS.gr=2000;
%                OPTIONS.Be=1;
%                OPTIONS.viamax=0.015;
%                OPTIONS.viamin=0.0075;
%                OPTIONS.tramax=0.3;
%                OPTIONS.tramin=0.2;
%                OPTIONS.tbemax=2;
%                OPTIONS.tsmax=2;
%                OPTIONS.fargs = {}
%
%
%Output:
%       Out : structure with three fields.
%             Out.V : matrix with all the viable parameter points found by
%                     the algorithm. Every vaible parameter point corresponds
%                     to a row of the matrix
%             Out.cost : column vector with the cost of the viable parameter
%                        points present in Out.V
%             Out.flag : structure with two fields.
%                Out.flag.vol : row vector with the volume covered by the
%                               enclosing ellipsoids in every iteration.
%                Out.flag.conv: boolean, true if the algorithm converged
%             Out.nfeval: number of function evaluations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~(size(x0,1)==1)
    error('MCEXP:IllegalArgument','Incorrect x0 dimension: must be a row vector representing a single viable parameter point.');
end


if ~(size(x0,2)==size(bmax,2) && size(x0,2)==size(bmin,2))
    error('MCEXP:IllegalArgument','x0, bmax, and bmin muss have the same column dimension.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OPTIONS = parseMCexpOptions(bmin,bmax, n, varargin{:});

dim=numel(x0);
if (n>=1)
    functionfinal('-s', threshold, n, dim, fun, OPTIONS.fargs);

    [viable, flagmonte, nfeval]=OEAMC(@functionfinal, x0, n, bmin,bmax, OPTIONS);

else

    viable = initviable(n,dim);
    flagmonte.vol = 0;
    flagmonte.conv = 0;

end

Out.V=viable(:,1:dim);
Out.cost=viable(:,dim+1);
Out.flag=flagmonte;
Out.nfeval=nfeval;

end



function optsout = parseMCexpOptions(bmin,bmax, n, varargin)
    % dim == numel(bmax) == numel(bmin);
    parser = inputParser;

    addParameter(parser, 'conv', false, @islogical);
    addParameter(parser, 'tol', 0.05, @isnonnegative);
    addParameter(parser, 'nint', 1e5, @isnonnegative); % does not use function evaluations
    addParameter(parser, 'gr', min(1000, max(10,floor(n/5))), @isnonnegative); % covariance adjustment frequency; originally was: 2000
    addParameter(parser, 'Be', 1, @isnonnegative);
    addParameter(parser, 'viamax', 0.015, @isnonnegative);
    addParameter(parser, 'viamin', 0.0075, @isnonnegative);
    addParameter(parser, 'tramax', 0.3, @isnonnegative);
    addParameter(parser, 'tramin', 0.2, @isnonnegative);
    addParameter(parser, 'tbemax', 2, @isnonnegative);
    addParameter(parser, 'tsmax', 2, @isnonnegative);
    % The square root of the determinant of the covariance matrix of
    % the multivariate normal distribution is the product of the standard
    % deviations of the principal components.
    % Initialise covariance matrix assuming uncorrelated dimensions
    % (cartesian axes being principal components) with std deviations of
    % 1/1000 of each of the given rectangular bounds.
    addParameter(parser, 'sigm', diag( ((bmax - bmin)/1000).^2 ), @isnonnegative);
    addParameter(parser, 'fargs', {}, @iscell);

    parse(parser, varargin{:});
    optsout = parser.Results;
end

function bool = isnonnegative(x)
    bool = isnumeric(x) && all(x(:) >= 0);
end
