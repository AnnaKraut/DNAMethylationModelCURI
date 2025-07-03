function tests = VolintTest()
%VOLINTTEST Unit tests for Volint function.
%
% Run with:
%   runtests('VolintTest')
%
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.TestData.plotSamples = ~isdebugging; % for 2-D problems
end

function teardownOnce(testCase)
end

%% Auxiliary
function X = unifsample(n,bmin,bmax)
    dim = numel(bmin);
    assert(dim == numel(bmax), 'Inconsistent dimensions of bounds.');
    bmin = bmin(:)'; bmax = bmax(:)';
    X = rand(n,dim); % samples from [0, 1]^dim
    for i=1:n % correct to [bmin, bmax]^dim
        X(i,:) = (bmax - bmin) .* X(i,:) + bmin;
    end
end

function [OutV] = VolintWrapper(V, cfunc, threshold, bmin, bmax, nsamp, plotSamples)
    tic;
    OutV = Volint(cfunc, threshold, V, bmax, bmin, nsamp);
    elapsed_time = toc;
    fprintf('Volint took %.2f seconds\n',elapsed_time);

    dim = size(V,2);
    if (dim==2) && plotSamples
        scatterellipse(V, OutV.V, OutV.cost, OutV.ellip.A, OutV.ellip.c,...
            'Original (black) and re-sampled and filtered points (colored by cost)');
    end
end

function verifyResults(testCase, OutV, exVol, exCi, tol)
    volRelErr = abs(OutV.vol - exVol)/exVol;
    volRelErrEst = OutV.err/exVol;
    fprintf('Exact volume rel. error: %.3f %%.\n',100*volRelErr);
    fprintf('Rel. estimated volume error of: %.3f %%.\n',100*volRelErrEst);
    assertTrue(testCase, volRelErr < tol, 'Volume wrong.');
    % Note: over-estimating error is fine, hence no abs(...)
    assertTrue(testCase, (volRelErr-volRelErrEst) < tol, 'Volume error estimate too low.');
    if exCi > 0
        ciRelErr = abs(OutV.ci - exCi)/exCi;
        ciRelErrEst = OutV.cierr/exCi;
        fprintf('Exact cost integral rel. error: %.3f %%.\n',100*ciRelErr);
        fprintf('Rel. estimated cost integral error of: %.3f %%.\n',100*ciRelErrEst);
    else
        ciRelErr = abs(OutV.ci);
        ciRelErrEst = OutV.cierr;
        fprintf('Exact cost integral abs. error: %.3f.\n',ciRelErr);
        fprintf('Abs. estimated cost integral error of: %.3f.\n',ciRelErrEst);
    end
    assertTrue(testCase, ciRelErr < tol, 'Cost integral wrong.');
    assertTrue(testCase, (ciRelErr-ciRelErrEst) < tol, 'Integral error estimate too low.');
end
%% Tests
function testSignprodSquare(testCase)
    cfunc = 'hstest.signprod';
    dim = 2;

    bmax = ones(1,dim);
    bmin = -bmax;
    nsamp0 = 5e3;


    name = sprintf('%d-D sign-product square: %d samples uniform bounding box.',dim, nsamp0);
    bstretch = 2;
    Vtot = unifsample(nsamp0, bstretch*bmin, bstretch*bmax);
    exVol = prod(bmax-bmin);
    exCi = 0;


    threshold = 1; % accept all within bounds
    nsamp = 5*nsamp0;


    fprintf('\n\n#### %s\n',name);

    OutV = VolintWrapper(Vtot, cfunc, threshold, bmin, bmax, nsamp, testCase.TestData.plotSamples);

    nsamp1 = size(OutV.V,1);
    fprintf('Requested/obtained samples ratio: %.2g\n', nsamp1/nsamp);
    % Volint filter bounds first
    assertTrue(testCase, (nsamp1/nsamp) > (0.9), 'Too few output samples.');
    tol = 0.05;
    fprintf('Error tolerance: %.2g\n', tol);
    verifyResults(testCase, OutV, exVol, exCi, tol);

    fprintf('\n\n');
end

function testTwoBalls(testCase)
    cfunc = 'hstest.twoBalls';
    dim = 2;

    radius = feval(cfunc,zeros(1,dim));
    bmax = (1+radius)*ones(1,dim);
    bmin = -bmax;
    nsamp0 = 1e3;


    name = sprintf('Two %d-Ball dummy: %d samples uniform bounding box.',dim, nsamp0);
    Vtot = unifsample(nsamp0, bmin, bmax);
    exVol = 2*volsph(dim)*radius^dim;
    exCi = 2*dim*volsph(dim)*radius^(dim+1)/(dim+1);


    threshold = radius;
    nsamp = 5*nsamp0;


    fprintf('\n\n#### %s\n',name);

    OutV = VolintWrapper(Vtot, cfunc, threshold, bmin, bmax, nsamp, testCase.TestData.plotSamples);

    nsamp1 = size(OutV.V,1);
    fprintf('Requested/obtained samples ratio: %.2g\n', nsamp1/nsamp);
    assertTrue(testCase, (nsamp1/nsamp) > (0.9)*(exVol)/(prod(bmax-bmin)), 'Too few output samples.');
    tol = 0.05;
    fprintf('Error tolerance: %.2g\n', tol);
    verifyResults(testCase, OutV, exVol, exCi, tol);

    fprintf('\n\n');
end

function testNBall(testCase)
    cfunc = @norm;
    radius = randi([2 10]);

    for dim=2:4
        bmax = radius*ones(1,dim); % dummy
        bmin = -bmax;

        for nlog=2:4
            nsamp0 = 10^nlog;

            % dummy uniform sample from the bounding box
            name = sprintf('%d-Ball dummy: %d samples uniform bounding box.',dim, nsamp0);
            % samples from [-radius, radius]^dim
            Vtot = unifsample(nsamp0, bmin, bmax);

            % expected values
            exVol = volsph(dim)*radius^dim;
            % Compute as 1-D integral over radius times area of (n-1) sphere of
            % that radius:
            %     Integral( area(S_{n-1}) * r^(n-1) * r, dr, radius, 0)
            % with:
            %     area(S_{n-1}) = n*Volume(S_{n})
            % giving:
            %     n*Volume(S_{n}) * r^(n+1)/(n+1) for r=radius
            %
            exCi = dim*volsph(dim)*radius^(dim+1)/(dim+1);

            % in case of, e.g., playing with scaling of the cost function, use:
            %threshold = funtest([zeros(1,dim-1) radius]);
            threshold = radius;
            nsamp = 5*nsamp0;


            fprintf('\n\n#### %s\n',name);

            OutV = VolintWrapper(Vtot, cfunc, threshold, bmin, bmax, nsamp, testCase.TestData.plotSamples);

            nsamp1 = size(OutV.V,1);
            fprintf('Requested/obtained samples ratio: %.2g\n', nsamp1/nsamp);
            assertTrue(testCase, (nsamp1/nsamp) > (0.9)*(exVol)/(prod(bmax-bmin)), 'Too few output samples.');
            tol = 50*(dim)/(nsamp0); % fairly safe, nothing formal
            fprintf('Error tolerance: %.2g\n', tol);
            verifyResults(testCase, OutV, exVol, exCi, tol);

            fprintf('\n\n');
        end
    end
end
