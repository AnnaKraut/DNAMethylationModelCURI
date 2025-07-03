function [a, A] = principaxesellip(E,d)
%PRINCIPAXESELLIP Given exes directions E of a d-dimensional ellipsoid,
%    compute lenghts of its semiprincipal-axes D and the principal axes W.
%
% Syntax:
%       [a, A] = principaxesellip(E,d)
%
% Inputs:
%       E = symmetric positive definite matrix describing the ellipsoid, i.e.:
%           (x-c)' E (x-c) = 1
%       d = leading dimensions of the ellipsoid to take into account
%
% Outputs:
%       a = lengths of ellipsoid principle axes
%       A = ellipsoid principle axes
%
    if (size(E,1) > d) || (size(E,2) > d)
        warning('PRINCIPAXESELLIP:ProblemReduction',...
            'Computing principal axes of an ellipsoid for a %d rank leading principal minor of the %d x %d ellipsoid matrix.',...
            d,size(E,1),size(E,2));
    end
    Eb = E(1:d,1:d);
    [~, Q, A] = svd(Eb);
    Qv = diag(Q)'; % Note: diag(Q) = [Q(j,j)]_j also for non-square matrices
    % length of the principle axes
    a = 1./sqrt(Qv);
end

% function str = mat2strml(m)
% %MAT2STRML Multi-line mat2str
% %
%     str = sprintf('\t%s',strrep(mat2str(m),';',sprintf(';...\n\t ')));
% end
