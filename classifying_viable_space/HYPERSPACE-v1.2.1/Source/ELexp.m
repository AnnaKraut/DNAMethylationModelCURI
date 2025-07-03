function Out = ELexp(fun,threshold,V,bmax,bmin,n,varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ELexp.m
%
%Obtains viable parameter points through the Multiple Ellipsoids Based
%Method
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Out = ELexp(function,threshold,V,bmax,bmin,n)
%   __  = ELexp(__,OPTIONS)
%
%Input:
%       function : function that recieves a parameter point and checks its cost.
%       threshold : scalar that defines the maximum value of the viable cost
%       V : matrix with viable parameter points. Every vaible parameter
%           point corresponds to a row of the matrix
%       bmax : row  vector with the upper bound of our parameter space
%       bmin : row vector with the lower bound of our parameter space
%       n : scalar that defines the maximum number of parameter
%       evaluations
%
%       OPTIONS : structure or name-value pairs with internal parameters of the
%                 algorithm or extra parameters for the cost function.
%                OPTIONS.tol : tolerance to accept the convergence
%                OPTIONS.grate : volume growth rate for each ellipsoid
%                                expansion attempt
%                OPTIONS.nexpmin : min. number of parameter points sampled
%                                  in each ellipsoid expansion attempt
%                OPTIONS.nint : number of ellipsoid cover integration
%                               points; sampled (not evaluated) to check
%                               the convergence
%                OPTIONS.fargs : cell with extra parameters for the cost
%                                function
%
%       DEFAULT VALUES:
%       ===============
%                OPTIONS.tol = 0.05
%                OPTIONS.grate = 2
%                OPTIONS.nexpmin = max(10,2*size(V,2))
%                OPTIONS.nint = 1e5
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
%             Out.flag : structure with two fields
%                Out.flag.vol : row vector with the volume covered by the
%                               enclosing ellipsoids in every iteration
%                Out.flag.conv : scalar equal to 1(0) if the algorithm
%                                converged (did not converge)
%             Out.nfeval: number of function evaluations
%             Out.ellip : structure describing found ellipsoids cover
%                         Out.ellip.A : (dim)x(dim)x(n elip) axes in cols
%                         Out.ellip.c : (dim)x(1)x(n elip) centers in cols
%                         Out.ellip.v : (n elip) volumes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if ~(size(V,2)==size(bmax,2) && size(V,2)==size(bmin,2))
    error('ELEXP:IllegalArgument','The elements of V, bmax, and bmin muss have the same column dimension.');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dim=size(V,2);

OPTIONS = parseELexpOptions(dim, varargin{:});

tol=OPTIONS.tol;
nexpmin=OPTIONS.nexpmin;
nint=OPTIONS.nint;
grate=OPTIONS.grate;
fargs=OPTIONS.fargs;

if (n>=1)

    functionfinal('-s', threshold, n, dim, fun, fargs);

%    if (size(V,1)>(2*dim+1))
%        V=cleaning(V(:,1:dim),dim);
%    end

    [viable, flagelip, nfeval, ellip]=MEBS(@functionfinal, V,dim,...
        n,nexpmin,nint, bmin,bmax, grate, tol);

else

    viable = initviable(n,dim);
    flagelip.vol = 0;
    flagelip.conv = 0;
    ellip.A = zeros(dim,dim,0);
    ellip.c = zeros(dim,1,0);
    ellip.v = [];

end

Out.V=viable(:,1:dim);
Out.cost=viable(:,dim+1);
Out.flag=flagelip;
Out.nfeval=nfeval;
Out.ellip=ellip;

end



function optsout = parseELexpOptions(d, varargin)
    parser = inputParser;

    addParameter(parser, 'tol', 0.05, @isnonnegative);
    addParameter(parser, 'nexpmin', max(10,2*d), @isnonnegative);
    addParameter(parser, 'nint', 1e5, @isnonnegative);
    addParameter(parser, 'grate', 2, @isgtone);
    addParameter(parser, 'fargs', {}, @iscell);

    parse(parser, varargin{:});
    optsout = parser.Results;
end

function bool = isnonnegative(x)
    bool = isnumeric(x) && all(x(:) >= 0);
end

function bool = isgtone(x)
    bool = isnumeric(x) && all(x(:) > 1);
end
