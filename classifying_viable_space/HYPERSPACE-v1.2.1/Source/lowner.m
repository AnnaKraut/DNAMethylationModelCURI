function [E, c]=lowner(a,tol)
% LOWNER Approximate Lowner ellipsoid.
%   [E C]=LOWNER(A,TOL) finds an approximation of the Lowner ellipsoid
%   of the points in the columns of A.  The resulting ellipsoid satisfies
%       x=A-repmat(C,size(A,2)); all(dot(x,E*x)<=1)
%   and has a volume of approximately (1+TOL) times the minimum volume
%   ellipsoid satisfying the above requirement.
%
%   A must be real and non-degenerate.  TOL must be positive.
%
%   Usually you can get faster results by using only the points on
%   the convex hull, e.g.:
%       [E c]=lowner(a(:,unique(convhulln(a'))),tol)
%
%   Example:
%       a=randn(2,100);
%       [E c]=lowner(a,0.001);
%       t=linspace(0,2*pi);
%       [V D]=eig(E);
%       e=repmat(c,size(t))+V*diag(1./sqrt(diag(D)))*[cos(t);sin(t)];
%       plot(a(1,:),a(2,:),'+',e(1,:),e(2,:),'-')
%
%   Reference:
%       Khachiyan, Leonid G.  Rounding of Polytopes in the Real Number
%           Model of Computation.  Mathematics of Operations Research,
%           Vol. 21, No. 2 (May, 1996) pp. 307--320.

%   See also: KHACHIYAN

%   Author: Anye Li (li.anye.0@gmail.com)
%   October 28, 2008

%   November 1, 2008:   Updated Khachiyan.
%                       Added an example.

[n,m]=size(a);

if n<1,
    error('LOWNER:IllegalArgument','To few rows in the input matrix; one or higher is required.');
end
% Not checked, since failing this should return 'KHACHIYAN:NonInvertibleMatrix'
% if m<n+1
%     error('LOWNER:IllegalArgument','To few columns in the input matrix; at least one more than nr of rows is required.');
% end

%   Find the Lowner ellipsoid of the centrally symmetric set lifted
%   to a hyperplane in a higher dimension.
F=khachiyan([a;ones(1,m)],tol);
%   Intersect with the hyperplane where the input points lie.
A=F(1:n,1:n); b=F(1:n,end);
c=-A\b; E=A/(1-c'*b-F(end));

% Force all the points to really be covered.
ac=a-repmat(c,1,m);
E=E/max(dot(ac,E*ac,1));
end
