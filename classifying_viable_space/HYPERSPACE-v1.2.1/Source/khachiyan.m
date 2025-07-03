function E=khachiyan(a,tol)
% KHACHIYAN Approximate Lowner ellipsoid of a centrally symmetric set.
%   E=KHACHIYAN(A,TOL) finds an approximation of the Lowner ellipsoid
%   of the points in the columns of [A -A].  The resulting ellipsoid
%   satisfies
%       all(dot(A,E*A)<=1)
%   and has a volume of approximately (1+TOL) times the minimum volume
%   ellipsoid satisfying the above requirement.
%
%   A must be real and non-degenerate.  TOL must be positive.
%
%   Reference:
%       Khachiyan, Leonid G.  Rounding of Polytopes in the Real Number
%           Model of Computation.  Mathematics of Operations Research,
%           Vol. 21, No. 2 (May, 1996) pp. 307--320.
%
%   See also: LOWNER
%
%   Author: Anye Li (li.anye.0@gmail.com)
%   October 28, 2008
%
%   November 1, 2008: Made the update equations more efficient.
%                     Fixed misleading comments.
%                     Made more sure that the ellipsoid really covers the
%                       points even after roundoff errors.
debugging = false;
%debugging = isdebugging;

[n, m]=size(a);
if n<2
    error('KHACHIYAN:IllegalArgument','To few rows in the input matrix - two or higher required.');
end
if ~isreal(a)
    error('KHACHIYAN:IllegalArgument','Non-real elements in the input matrix.');
end
if ~(isreal(tol) && tol>0)
    error('KHACHIYAN:IllegalArgument', 'Non-positive tolerance.');
end
%   Initialize the barycentric coordinate descent.
invA=m*inv(a*a');
if any(~isfinite(invA(:)))
    error('KHACHIYAN:NonInvertibleMatrix', 'Failed to invert A*A''.');
end
w=dot(a,invA*a,1);
%   Khachiyan's BCD algorithm for finding the Lowner ellipsoid of a
%   centrally symmetric set.
%   Refer to (2.7),(2.10),(2.18),(BCD), and the end of Lemma 4 for
%   the iteration used here.  The variable names are basically the
%   same, except f, g, and h which are common factors.
%   Read the paper if you want to actually understand how it works.
iBCD = 1;
iBCDMax = 1e5; % maximum number of iterations (max observed < 1e4)
while 1
    [w_r, r]=max(w);
    f=w_r/n; epsilon=f-1; % @MR, TODO: check if f-1 < epsilon or rather |f-1| < epsilon (e.g. when inverse fails)?
    if (epsilon<=tol)
        if debugging % report finish
            fprintf('[DEBUG] khachiyan, BCD algorithm, converged after iteration #%d, with epsilon = %f <= tol = %f.\n',iBCD,epsilon,tol);
        end
        break;
    end
    iBCD = iBCD + 1;
    if (iBCD > iBCDMax)
        warning('KHACHIYAN:MaxIter','failed to converge in %d iterations.',iBCDMax);
        if debugging % report finish
            fprintf('[DEBUG] khachiyan, BCD algorithm, failed to converge in %d iterations, with epsilon = %f > tol = %f.\n',iBCDMax,epsilon,tol);
        end
        break;
    end
    if debugging && (mod(iBCD,1000) == 1) % report progress every 1000 steps
        fprintf('[DEBUG] khachiyan, BCD algorithm, iterations #%d .. %d, started with epsilon = %f > tol = %f.\n',iBCD,iBCD+1000-1,epsilon,tol);
    end
    g=epsilon/((n-1)*f); h=1+g; g=g/f;
    b=invA*a(:,r); invA=h*invA-g*b*b';
    bTa=b'*a; w=h*w-g*(bTa.*bTa);
end
E=invA/w_r;
% Accumulated roundoff errors may cause the ellipsoid
% to not quite cover all the points.
% Use
%   E=invA/max(dot(a,invA*a,1));
% if you don't like that.
end
