function [OutV, flags, nfeval] = OEAMC(funcion, x0, niter, bmin,bmax, OPTIONS)
%OEAMC Carry out the out-of-equilibrium adaptive Markov Chain Monte Carlo
% exploration of the parameter space.
%
% Semantics:
%      [OutV, flags, nfeval] = OEAMC(funcion, x0, n, bmin,bmax, OPTIONS)
%
%
% Inputs:
%       funcion = function that recieves a parameter point and
%               returns two variables. The first flag is true(false)
%               if the parameter point is viable (nonviable).
%               The second is the value of the cost function for this
%               parameter point.
%
%       x0 = initial parameter point of the exploration.
%
%       n = maximum number of parameter point evaluations.
%
%       bmin = vector with the lower bound of our parameter space.
%       bmax = vector with the upper bound of our parameter space.
%
%       OPTIONS = structure with internal parameters of the algorithm.
%                OPTIONS.conv : boolean, if to check the convergence
%                               (default: false)
%                OPTIONS.tol = tolerance to accep the convergence
%                OPTIONS.nint = points sampled in order to check the convergence
%                OPTIONS.gr = number of parameter evaluations among updates
%                OPTIONS.Be = initial value of \Beta;
%                OPTIONS.viamax = maximum frequency of viable points that maintain constant \Beta
%                OPTIONS.viamin = minimum frequency of viable points that maintain constant \Beta
%                OPTIONS.tramax = maximum frequency of accepted transitions to maintain constant the covariance matrix
%                OPTIONS.tramin = minimum frequency of accepted transitions to maintain constant the covariance matrix
%                OPTIONS.tbemax = maximum relative change of \Beta among updates
%                OPTIONS.tsmax = maximum relative change of the covariance
%                                matrix among updates
%
% Outputs:
%       OutV = matrix of all the viable paremeter points found by the algorithm
%              and their cost in the last column
%       flags = structure with two fields.
%               flags.vol  = volume covered by the enclosing ellipsoids
%                            in every iteration.
%               flags.conv = true(false) if the algorithm converged
%                            (it did not converge)
%       nfeval = number of function evaluations
%
debugging = isdebugging();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

assert(niter>=1);
assert(isvector(x0));
dim=numel(x0); %dimension of the parameter space

nfailupdates=0; % number of updates without finding a single viable point.
gr=OPTIONS.gr;% number of iterations among updates (temp, covariance, opt. MVEE)

checkConverged=OPTIONS.conv; % if to check convergence
converged=false; % if converged
tol=OPTIONS.tol;% tolerance to accept convergence
ni=OPTIONS.nint;% points used to calculate MVEEs volume
[~, cmincount] = nclusmax(1, dim, 1); % min nr of points per MVEE
nstep=min(5*max(cmincount, gr),floor(niter/10)); % number of parameter evaluations among two convergence tests
ntol=max(3,floor(niter/nstep/10));% number of previous iterations to check convergence against
EjesA = []; % MVEEs for checking convergence

naccepted=0; % number of accepted proposals
cont=0; %Stores the number of viable parameters among updates
nrestarts=0; %Stores the number of restarts.
Volexp=zeros(1,0);%Stores the volume of the ellipsoids in the different iterations

tbemax=OPTIONS.tbemax;%maximum relative change of \Beta among updates
tsmax=OPTIONS.tsmax;%maximum relative change of the covariance matrix among updates

viamax=OPTIONS.viamax;% maximum frequency of viable points that maintain constant \Beta
viamin=OPTIONS.viamin;% minimum frequency of viable points that maintain constant \Beta

tramax=OPTIONS.tramax;% maximum frequency of accepted transitions to maintain constant the covariance matrix
tramin=OPTIONS.tramin;% minimum frequency of accepted transitions to maintain constant the covariance matrix


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% \Beta and \Sigma initialization    %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Be=OPTIONS.Be; %initial value of \Beta
sigm=OPTIONS.sigm;%initial value of the covariance matrix



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Markov Chain Metropolis%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxiterx0 = 1e1 ; % max attempts for finding viable points between temperature updates

nfeval=1;
[DIA, A]=feval(funcion,x0);

while ~converged && (nfeval<niter)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Sample proposal %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Choose a new parameter y0 around the previous x0
    [y0, sigm] = generateSample(x0,sigm,bmin,bmax);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Test proposal %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Evaluates the new parameter
    [DIB, B]=feval(funcion,y0);

    cont=cont+DIB;
    nfeval=nfeval+1;

    %%%% Metropolis acceptance criteria
    % Accept the new point if the cost function is smaller
    if (B<A)
        x0=y0;
        A=B;
        naccepted=naccepted+1;
    % If it is larger, compare it with the Boltzman distribution
    elseif (exp(-Be*(B-A))>rand)
        x0=y0;
        A=B;
        naccepted=naccepted+1;
    % If it is smaller we keep the old point
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Update the temperature and the covariance matrix %%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Condition to update
    if (mod(nfeval,gr)==0)

        viable = feval(funcion,'-g');
        U=nviablepts(viable); %Total number of viable points

        %%% Update clustering and MVEEs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if checkConverged
            if isempty(EjesA)
                Vtotaldef=viable(1:U,1:dim); % Viable points found from the start
            else
                V=viable(U1:U,1:dim); % Viable points found since the last updates.

                % find viable points lying outside of the previous MVEE
                for iellip=1:numel(Volumenelip)
                    V=ellipdiff(V,EjesA(:,:,iellip),centccc(:,:,iellip),dim);
                end

                % append only the outside points
                lv=length(V(:,1));
                Vtotaldef(lvtd+1:(lvtd+lv),:)=V;
            end

            % try to cluster the viable points and compute MVEEs; skip if failed
            if size(Vtotaldef,1)>cmincount % there is enough points
                try
                    [EjesA, centccc, Volumenelip]=coverellipsoid(Vtotaldef,dim);
                    % update indices for the next update
                    U1=U;
                    lvtd=size(Vtotaldef,1);
                catch ME
                    if strcmp(ME.identifier,'CLUSTERIZE:SingularMVEE')
                        warning('OEAMC:SingularMVEE','Failed to compute initial minimal volume enclosing ellipsoids using the current sample.');
                    else
                        rethrow(ME);
                    end
                end

                if debugging
                    fprintf('[DEBUG] OEAMC, Total number of viable parameter points found = %d.\n',U);
                    fprintf('[DEBUG] OEAMC, Number of parameter points necessary to define the enclosing ellipsoids = %d.\n',size(Vtotaldef,1));
                end
            end
        end



        %%% Updating the temperature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if debugging
            fprintf('[DEBUG] OEAMC, Total number of parameters points checked = %d.\n',nfeval);
            fprintf('[DEBUG] OEAMC, Frequency of accepted transitions = %.3g.\n',naccepted/gr);
            fprintf('[DEBUG] OEAMC, Frequency of viable parameters points = %.3g.\n',cont/gr);
            fprintf('[DEBUG] OEAMC, Old values of: \\Beta = %.3g, \\Sigma = %s.\n',Be,mat2str(diag(sigm)));
            fprintf('[DEBUG] OEAMC, Number of restarts = %d.\n',nrestarts);
        end

        % if the frequency of viable points is higher than the threshold
        if (cont>(gr*viamax))

            tasa=tbemax*(cont-gr*viamax)/cont; % proportional increase of the temperature
            Be=Be/(1+tasa);
            nfailupdates=0; %number of updates without viable points

        elseif (cont<=(gr*viamin))

            % If it did not find any viable point between updates
            if cont==0

                nfailupdates=nfailupdates+1;

                % only updates the temperatures if carries out a minimum
                % number of transitions
                if naccepted>gr*tramin

                    tasa=tbemax;
                    Be=Be*(1+tasa);

                end


                % If it is lost, choose a viable point to restart the search
                if nfailupdates>maxiterx0
                    if U > 1
                        warning('OEAMC:MaxIter','Reached maximum number of iterations without finding a viable point between updates, re-starting from a different random viable point.\n');
                        ch = randi([1,U]);
                        x0new = viable(ch,1:dim);
                        if (x0new == x0) % same point => pick next
                            ch = mod(ch,U)+1;
                            x0 = viable(ch,1:dim);
                        else
                            x0 = x0new;
                        end
                    else
                        error('OEAMC:MaxIter','Reached maximum number of iterations without finding a viable point between updates and there is no other viable point to re-start from.\n');
                    end

                    Be=Be*tbemax;
                    nrestarts=nrestarts+1;
                end

            else

                tasa=tbemax*(gr*viamin-cont)/(gr*viamin);
                Be=Be*(1+tasa);
                nfailupdates=0;

            end

        end



        %%%% Updating the covariance matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TODO Consider adaptive local covariance using the ellipsoids
%        % if x0 belongs to some ellipsoid, use it as a local covariance matrix
%        if ~isempty(EjesA)
%            if numel(size(EjesA)<3), nellip = 1;
%            else, nellip = size(EjesA,3); end
%            isel = false; iellip = 0;
%            while ~isel & (iellip < nellip)
%                iellip = iellip + 1;
%                isel = isempty(ellipdiff(x0,EjesA(:,:,iellip),centccc(:,:,iellip),dim));
%            end
%            if isel, sigm = lintransfsph(EjesA(:,:,iellip),dim)/4; end
%            % /4 because 2*SD = sqrt(4*Var) covers ca. 96% of Norm dist
%        end

        % grow or shrink covariance matrix
        if (naccepted>gr*tramax)&&(cont>0)
            actionStr='growing';
            tasa=tsmax*(naccepted-gr*tramax)/naccepted;
            sigm=sigm*(1+tasa);
        elseif (naccepted<gr*tramin)
            actionStr='shrinking';
            tasa=tsmax*(gr*tramin-naccepted)/(gr*tramin);
            sigm=sigm/(1+tasa);
        else
            actionStr='';
        end
        if debugging && ~isempty(actionStr)
            fprintf('[DEBUG] OEAMC, %s proposal covariance by %.3g.\n', actionStr, 1+tasa);
        end

        naccepted=0;
        cont=0;


        if debugging
            fprintf('[DEBUG] OEAMC, New values of: \\Beta = %.3g, \\Sigma = %s.\n',Be,mat2str(diag(sigm)));
            if checkConverged
                fprintf('[DEBUG] OEAMC, Enclosed volumes in sub-sequent iterations = %s.\n',mat2str(Volexp));
            end
        end

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Check convergence to stop the simulation %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (mod(nfeval,nstep)==0)
        fprintf('[INFO] OEAMC, progress: %2.1f%%.\n', 100*(nfeval/niter));
    if checkConverged

        m=nfeval/nstep;
        Volexp(m)=0;

        viable = feval(funcion,'-g');
        U = nviablepts(viable);


        if isempty(EjesA)
            Vtotaldef=viable(1:U,1:dim); % viable points found from the start
        else
            V=viable(U1:U,1:dim); % viable points found from the last check
            % find viable points lying outside of the previous MVEE
            for iellip=1:numel(Volumenelip)
                V=ellipdiff(V,EjesA(:,:,iellip),centccc(:,:,iellip),dim);
            end
            % append only the outside points
            lv=length(V(:,1));
            Vtotaldef(lvtd+1:(lvtd+lv),:)=V;
        end

        if size(Vtotaldef, 1) > cmincount
            [ Volexp(m), ~, ~, ~, EjesA, centccc, Volumenelip ] = volestellip(Vtotaldef,dim, ni, bmin,bmax, 1);
            %Vtotaldef=cleaning(Vtotaldef,dim);
            U1=U;
            lvtd=size(Vtotaldef,1);
        end

        % Check condition to stop the exploration (converged): relative
        % change of MVEE total volume < tolerance
        if (m>ntol)
            [converged, relVolDelta] = conver(Volexp(1:m), ntol+1, tol);
            if debugging
                fprintf('[DEBUG] OEAMC, max. relative volume change over last %d progress checks: %2.3f%%.\n', ntol, 100*max(relVolDelta));
            end
        end

    end
    end

end

[viable, nfevalActual] = feval(funcion,'-g');
assert(nfevalActual == nfeval, 'Oops, wrong count of function evaluations.');
U = nviablepts(viable);
OutV=viable(1:U,:);

flags.vol = Volexp;
flags.conv = converged;

if checkConverged
    if converged, convStr = 'converged';
    else convStr = 'did not converge';
    end
else convStr = 'finished w/o convergence check';
end
fprintf('[INFO] OEAMC, %s in %d iterations; found %d viable samples in %d function evaluations (given max. %d).\n',convStr,nfeval, U, nfeval, niter); % i = while loop iterations

end



function [y0, sigm] = generateSample(x0,sigm,bmin,bmax)
debugging = isdebugging;

    maxiter = 1e4; % max sampling attempts for a new point within bounds
    buffsize = 1e3; % buffer size for mvnrnd sampling
    % If y0 does not fit the bounds, sample until it does, periodically
    % narrowing proposal distribution.
    niter = 0;
    inbound = false;
    while ~inbound && (niter < maxiter)
        buffy0 = mvnrnd(x0,sigm,buffsize);
        idxy0 = find(~isoutofbound(buffy0,bmin,bmax),1);
        inbound = ~isempty(idxy0);
        if ~inbound % relax proposal covariance every y0buffsize tries
            warning('OEAMC:LargeCov','The elements of the new sample covariance matrix are reduced due to countour problems.\n');
            sigm=sigm/10;
            niter = niter + buffsize;
        else
            y0 = buffy0(idxy0,:);
            niter = niter + idxy0;
        end
    end
    if debugging && (niter > 1) % if there was re-sampling, report how much
        fprintf('[DEBUG] OEAMC>generateSample, finished after #%d tries.\n',niter);
    end
    if (niter >= maxiter)
        error('OEAMC:MaxIter','Failed to find a sample within given bounds (in %d tries around x0=%s).',maxiter, mat2str(x0));
    end

end
