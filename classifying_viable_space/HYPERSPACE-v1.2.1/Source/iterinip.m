function [V, fv] = iterinip(funcion, k0,dim, bmin,bmax)
%ITERINIP Find points near the border of the viable volume, in the
% direction of cartesian axes, starting at k0.
%
% Sytax:
%     V = iterinip(funcion, k0,dim, bmin,bmax)
%     [V, fv] = iterinip(___)
%
% Inputs:
%     funcion = indicator (and cost) function to check if a given point is
%               viable
%
%     k0 = initial viable point to expand ellipsoid around
%     dim = dimension of the points
%
%     bmin = vector with the lower bounds for samples
%     bmax = vector with the upper bounds for samples
%
% Outputs:
%     V = 2*dim x dim matrix with 2*dim points inside the viable space,
%         positioned in the direction of the cartesian axes centered in k0
%     fv = values of the function for points in V
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 2*dim; % nr of viable points: +/- direction in each dimension
bounds = [bmin; bmax];
max_retry = 100; % nr of max. re-tries for a single point

V=zeros(n,dim);
fv = nan(n,1);
tol=0.1; % was: 0.2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Finding viable points in the cartesian axes %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TODO add random rotation of the cartesian axes to avoid fixed counterexamples
%     R = orth(randn(dim,dim))
%     for i=1:(2*dim)
%         oxi = ...
%         oxi_rot = R * oxi
%         sectioning between k0 and bound in the oxi_rot direction
%         (cf. hs2.mcexp.XRayShoot function)
for i=1:n

    P=k0;
    idim = ceil(i/2);

    xi_in=k0(idim);
    xi_out=bounds(mod(i,2)+1,idim); % take subsequently bmax(ix), bmin(ix), bmax(ix+1), ...

    %err=abs(xi_in-xi_out)/min(abs(xi_in),abs(xi_out));
    P(idim)=xi_out;
    [A, cost]=feval(funcion,P);

    if ~A

        j=0; retry_success = false;
        while (j < max_retry) && ~retry_success;
            j=j+1;
            % not a success (err>=tol) but good enough
            if (j>=ceil(max_retry/2))&&(xi_in ~= k0(idim)),
                break
            end

            % evaluate new proposal
            x = (xi_out+xi_in)/2;
            P(idim) = x;
            [A, cost] = feval(funcion,P);

            if (A==1)
                % Rem: now worries, all viable points are saved directly in
                % functionfinal - no waste there
                xi_in=x;
            else
                xi_out=x;
            end

            err = abs(xi_in-xi_out)/min(abs(xi_in),abs(xi_out));
            retry_success = (err<tol) && (xi_in ~=k0(idim));
        end

        % on failure: warn and exit
        if ~retry_success && (j>=max_retry)
            warning('ITERINIP:MaxIter', 'Reached maximum number of sections in a direction of dimension %d without finding a viable point.', idim);   
            V=[]; fv=[];
            return
        end

        P(idim)=xi_in;
    end

    V(i,:)=P;
    fv(i)=cost;

end


end
