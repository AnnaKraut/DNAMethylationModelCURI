function tests = AllTest()
%VOLINTTEST Unit tests for MCexp+ELexp+Volint pipeline.
%
% Run with:
%   runtests('AllTest')
%
    tests = functiontests(localfunctions);

%     tests = functiontests({@setupOnce, @teardownOnce, @setup, @teardown,...
%         @testIntegrateDoubleDonut});
%         @testIntegrate});
%         @testMCexpConvergence});
%         @testIntegrateDonut});
%         @testIntegrateWurst});
%         @testIntegrateDonut, @testIntegrateDoubleDonut, @testIntegrateWurst});

end



function setupOnce(testCase)


    % bulk integration: test setup
    testCase.TestData.plotSamples = ~isdebugging; % for 2-D problems
    s.nfeval_vec = 5e4; % nr of samples
    s.dim_vec = [2 4]; % nr of dimensions
    s.volRelErrTol = 0.05; % allowed volume relative error (0 = not asserted)
    s.nrep = 1; % number of repeats
    s.parallelize = false; % over repeats
    s.saveMat = false;
    s.nfevalProportionsArray = [...
        3 6 1
    ]; % function evaluations proprotions for MCexp, ELexp and Volint


%     % bulk integration: quick benchmarks setup
%     testCase.TestData.plotSamples = ~isdebugging;
%     s.dim_vec = 10; % [10 15];
%     s.nfeval_vec = 2e5; %[1e5 2e5];
%     s.volRelErrTol = 0;
%     s.nrep = 1;
%     s.saveMat = true;
%     s.parallelize = true; % over repeats
%     s.nfevalProportionsArray = [...
%         4 5 1; % 2 6 2;
%     ]; % function evaluations proprotions for MCexp, ELexp and Volint

%     % bulk integration: full benchmarks setup
%     testCase.TestData.plotSamples = false;
%     s.dim_vec = [2 5 10 15];
%     s.nfeval_vec = [5e4 1e5 2e5 5e5];
%     s.volRelErrTol = 0;
%     s.nrep = 4;
%     s.saveMat = true;
%     s.parallelize = true; % over repeats
%     s.nfevalProportionsArray = [...
% %         4 3 3;...
% %         3 4 3;...
%         4 5 1;...
% %         3 5 2;...
%         3 6 1;%...
% %         2 7 1;
%     ]; % function evaluations proprotions for MCexp, ELexp and Volint

    testCase.TestData.bulk = s;
end

function teardownOnce(testCase)
end

function setup(testCase)
end

function teardown(testCase)
end



%% Auxiliary
function [OutV, nfeval, nsec] = integrate(testCase, costfun, threshold, x0, bmin, bmax, nfevalSteps, fargs)
    elapsed_time = inf(3,1);

    tic;
    Vm = MCexp(costfun,threshold,x0,bmax,bmin,nfevalSteps(1),'fargs',fargs);
    elapsed_time(1) = toc;
    tic;
    Ve = ELexp(costfun,threshold,Vm.V,bmax,bmin,nfevalSteps(2),'fargs',fargs);
    elapsed_time(2) = toc;
    Vtot = vertcat(Vm.V,Ve.V);
    tic;
    OutV = Volint(costfun, threshold, Vtot, bmax, bmin, nfevalSteps(3),'fargs',fargs);
    elapsed_time(3) = toc;

    nfeval = Vm.nfeval + Ve.nfeval + OutV.nfeval;
    nsec = sum(elapsed_time);

    fprintf('MCexp+ELexp+Volint, time: %.2f seconds.\n', nsec);
    fprintf('MCexp+ELexp+Volint, nfeval: %d (given: %d)\n', nfeval, sum(nfevalSteps));

    if (numel(x0)==2) && testCase.TestData.plotSamples
        scatterellipse(Vtot, OutV.V, OutV.cost, OutV.ellip.A, OutV.ellip.c,...
            'Original (black) and re-sampled and filtered points (colored by cost)');
    end

    % some extra info
    fprintf('\n');
    fprintf('MCexp took %.2f seconds\n',elapsed_time(1));
    fprintf('ELexp took %.2f seconds\n',elapsed_time(2));
    fprintf('Volint took %.2f seconds\n',elapsed_time(3));
    fprintf('\n');
    fprintf('Volint, requested/returned samples ratio: %.2g\n', size(OutV.V,1)/nfevalSteps(3));

    fprintf('\n\n');

end

function [volRelErr, volRelErrEst] = checkVolume(testCase, OutV, exVol, tol)
    volRelErr = abs(OutV.vol - exVol)/exVol;
    volRelErrEst = OutV.err/exVol;
    fprintf('Exact volume rel. error: %.3f %%.\n',100*volRelErr);
    fprintf('Rel. estimated volume error of: %.3f %%.\n',100*volRelErrEst);
    if tol > 0
        assertTrue(testCase, volRelErr < tol, sprintf('Volume wrong (with tol=%.3g).', tol));
        % Note: over-estimating error is fine, hence no abs(...)
        assertTrue(testCase, (volRelErr-volRelErrEst) < tol, 'Volume error estimate way too low.');
    end
    fprintf('\n\n');
end

function [numWorkers, rndSeed, rndOffsetList] = initParallel(parallelize, max_n)
    numWorkers = 0;
    if parallelize && max_n > 1
        try
            ppool = gcp; % will create parallel pool if one hasn't been created yet
            numWorkers = ppool.NumWorkers;
            assert(isposint(numWorkers));
        catch ME
            warning('Parallel Computing Toolbox not found.\nThe follwoing error was caught while calling gcp:\n\t%s: %s\n',ME.identifier, ME.message);
        end
    end
    rng('shuffle')
    rndSeed = RandStream.getGlobalStream.Seed;
    rndOffsetList = randi(max_n,1,1e5);
end

function printTestHeader(name, varargin)
    hl = repmat('#',1,80);
    fprintf('\n%s\n', hl);
    printTestSubheader(name, varargin{:});
end
function printTestSubheader(name, varargin)
    fprintf('#### %s\n', sprintf(name, varargin{:}));
end

function [OutVArray, nfevalArray, nsecArray, volRelErrArray] = integrateBulk(testCase, name,...
    dim_vec, nfeval_vec, nfevalProportions, nrep, parallelize,...
    costfun_ca, threshold_vec, x0_ca, bmin_ca, bmax_ca, fargs_ca,...
    vol_vec, volRelErrTol, saveMat)
%INTEGRATEBULK ...
%
    printTestHeader(name);

    % tolerance for the exact volume relative error
    if isempty(volRelErrTol)
        volRelErrTol = 0;
    end

    % function evaluations proprotions for MCexp, ELexp and Volint
    if isempty(nfevalProportions)
        nfevalProportions = [4 2 4];
    end
    nfevalProportions = nfevalProportions/sum(nfevalProportions);

    for i_dim=numel(dim_vec):-1:1
        dim = dim_vec(i_dim);
        exVol = vol_vec(i_dim);

        % cost function and threshold
        if iscell(costfun_ca), costfun = costfun_ca{i_dim};
        else costfun = costfun_ca; end
        if numel(threshold_vec) > 1, threshold = threshold_vec(i_dim);
        else threshold = threshold_vec; end

        % x0 and the bounding box
        x0 = x0_ca{i_dim};
        bmin = bmin_ca{i_dim};
        bmax = bmax_ca{i_dim};

        % extra cost function args
        if ~isempty(fargs_ca)
            % args per dimension
            if size(fargs,1)>1, fargs = fargs_ca{i_dim,:};
            % args common for all dimensions
            else fargs = fargs_ca; end
            % no args
        else fargs = {}; end

        [numWorkers, rndSeed, rndOffsetList] = initParallel(parallelize, nrep);

        % loop over nr of function evaluations:
        for i_nfeval = numel(nfeval_vec):-1:1,
            nfeval0 = nfeval_vec(i_nfeval);
            nfevalSteps = splitlrm(nfeval0,  nfevalProportions);

            OutV_ca = cell(1, nrep);
            temp_volRelErr_vec =  nan(1, nrep);
            temp_nfeval_vec = nan(1, nrep);
            temp_nsec_vec = nan(1, nrep);
            parfor (irep=1:nrep, numWorkers)
                try
                    printTestSubheader('dim=%d, nfeval=%d, rep #%d', dim, nfeval0, irep);
                    rng(rndSeed+rndOffsetList(irep));
                    [OutV, nfeval, nsec] = integrate(testCase, costfun, threshold, x0, bmin, bmax, nfevalSteps, fargs);
                    volRelErr = checkVolume(testCase, OutV, exVol, volRelErrTol);
                    OutV_ca{irep} = OutV;
                    temp_volRelErr_vec(irep) = volRelErr;
                    temp_nfeval_vec(irep) = nfeval;
                    temp_nsec_vec(irep) = nsec;
                catch ME
                    if strcmp('MATLAB:unittest:Assertable:AssertionFailed', ME.identifier)
                        rethrow(ME);
                    else
                        fprintf('[ERROR] %s: %s\n', ME.identifier, ME.message);
                    end
                end
            end
            for irep=nrep:-1:1
                OutVArray(i_dim,i_nfeval,irep) = OutV_ca{irep};
            end
            volRelErrArray(i_dim,i_nfeval,:) = temp_volRelErr_vec;
            nfevalArray(i_dim,i_nfeval,:) = temp_nfeval_vec;
            nsecArray(i_dim,i_nfeval,:) = temp_nsec_vec;
        end
    end

    % for plotting later
    if saveMat
        results_fn = sprintf('test_%s_%s.mat',name, datestr(now, 'yymmdd_HHMMSS'));
        save(results_fn, 'OutVArray', 'volRelErrArray', 'nfevalArray', 'nsecArray', 'vol_vec', 'dim_vec', 'nfeval_vec', 'nfevalProportions');
    end

end



%% Tests
function testIntegrate(testCase)
%TESTINTEGRATE Just run and inspect visually - no assertions.

    costfun = 'hstest.twoBalls';
    center = 0.5;
    fargs = {center};
    dim = 2;

    % disjoint spheres
    %radius = feval(costfun,zeros(1,dim),center);
    %for assertions:
    %exVol = 2*volsph(dim)*radius^dim;
    %exCi = 2*dim*volsph(dim)*radius^(dim+1)/(dim+1);
    % overlaping spheres
    radius = 1.5*sqrt(2*center^2);

    bmax = (center+radius)*ones(1,dim);
    bmin = -bmax;
    nsamp0 = 1e2;

    threshold = radius;

    x0 = zeros(1,dim);
    nfevalSteps = [2*nsamp0 5*nsamp0 3*nsamp0];

    printTestHeader('testIntegrate: %s, dim=%d, nfeval=%d', costfun, dim, sum(nfevalSteps));
    OutV = integrate(testCase, costfun, threshold, x0, bmin, bmax, nfevalSteps, fargs);

end


function testMCexpConvergence(testCase)
%TESTMCEXPCONVERGENCE Test convergence of MCexp, which in the default
% integration workflow is not checked.
%

    costfun = 'hstest.twoBalls';
    center = 0.5;
    fargs = {center};
    dim = 5;

    % disjoint spheres
    radius = feval(costfun,zeros(1,dim),center);
    volRelStr = '=';
%     % overlaping spheres
%     radius = 1.5*sqrt(2*center^2);
%     volRelStr = '<';

    % exact volume of two spheres
    exVol = 2*volsph(dim)*radius^dim;

    bmax = 10*(center+radius)*ones(1,dim);
    bmin = -bmax;
    nfeval = 1e5;

    threshold = radius;

    x0 = zeros(1,dim);

    printTestHeader('testMCexpConvergence: %s, dim=%d, nfeval=%d', costfun, dim, nfeval);
    tic;
    Vm = MCexp(costfun,threshold,x0,bmax,bmin,nfeval,'fargs',fargs,'conv',true,'tol',0.1);
    elapsed_time = toc;
    fprintf('MCexp took %.2f seconds\n',elapsed_time);
    fprintf('Volume to cover %s %.3g.\n', volRelStr, exVol);
    fprintf('MVEEs volume over iterations: %s.\n', mat2str(Vm.flag.vol,3+max(-floor(log10(exVol)),0)));

    assertTrue(testCase, Vm.flag.conv,...
        sprintf('MCexp was expected to converge in %d function evaluations (dim=%d).',Vm.nfeval, dim));

    if (dim==2)
        scatterellipse([], Vm.V, Vm.cost, zeros(dim,dim,0), zeros(dim,1,0),...
            'MCexp points (colored by cost)');
    end
end


function testIntegrateDonut(testCase)
%TESTINTEGRATEDONUT Single shell example from the original paper.
% Beware: runs very long (ca. 4x2min).
%
    testName = 'Donut';

    % bulk integration setup
    nfeval_vec = testCase.TestData.bulk.nfeval_vec;
    dim_vec = testCase.TestData.bulk.dim_vec;
    volRelErrTol = testCase.TestData.bulk.volRelErrTol;
    nrep = testCase.TestData.bulk.nrep;
    parallelize = testCase.TestData.bulk.parallelize;
    nfevalProportionsArray = testCase.TestData.bulk.nfevalProportionsArray;
    saveMat = testCase.TestData.bulk.saveMat;

    % cost function and the threshold
    r_small = 0.3;
    r_large = 0.5;
    costfun = @(x) hstest.shell(x, r_small, r_large); % alt: use fargs per dim
    threshold = (r_large-r_small)/2;

    % setup per dimension
    for dim_idx=numel(dim_vec):-1:1
        dim = dim_vec(dim_idx);
        % exact volume
        vol_vec(dim_idx) = 1*volsph(dim)*(r_large^dim - r_small^dim);
        % initial viable point
        x0 = zeros(1,dim); x0(randi([1 dim])) = r_small + 2*rand*threshold;
        assertTrue(testCase, costfun(x0) <= threshold, 'Blah! Smth wrong w/ x0 or the cost function.');
        x0_ca{dim_idx} = x0;
        % bounds
        bmax_ca{dim_idx} = 10*ones(1,dim);
        bmin_ca{dim_idx} = -10*ones(1,dim);
    end

    for i=1:size(nfevalProportionsArray,1)
        nfevalProportions = nfevalProportionsArray(i,:);
        name = sprintf('%s_%s', testName, regexprep(int2str(nfevalProportions),'\s',''));
        integrateBulk(testCase, name,...
            dim_vec, nfeval_vec, nfevalProportions, nrep, parallelize,...
            costfun, threshold, x0_ca, bmin_ca, bmax_ca, {},...
            vol_vec, volRelErrTol, saveMat);
    end

end

function testIntegrateDoubleDonut(testCase)
%TESTINTEGRATEDOUBLEDONUT Tangent shell example from the original paper.
% Beware: runs very long (ca. 4x2min).
%

    testName = 'DoubleDonut';

    % bulk integration setup
    nfeval_vec = testCase.TestData.bulk.nfeval_vec;
    dim_vec = testCase.TestData.bulk.dim_vec;
    volRelErrTol = testCase.TestData.bulk.volRelErrTol;
    nrep = testCase.TestData.bulk.nrep;
    parallelize = testCase.TestData.bulk.parallelize;
    nfevalProportionsArray = testCase.TestData.bulk.nfevalProportionsArray;
    saveMat = testCase.TestData.bulk.saveMat;

    % cost function and the threshold
    r_small = 0.3;
    r_large = 0.5;
    tp_dim = 1; % tangent point dimension
    costfun = @(x) hstest.shellsTangent(x, r_small, r_large, tp_dim); % alt: use fargs per dim
    threshold = (r_large-r_small)/2;

    % setup per dimension
    for dim_idx=numel(dim_vec):-1:1
        dim = dim_vec(dim_idx);
        % exact volume
        % e.g. for dim=10, 2*2.5502*9.7066e-04 = 2*2.5e-3 = 5e-3
        vol_vec(dim_idx) = 2*volsph(dim)*(r_large^dim - r_small^dim);
        % initial viable point: furthest from the tangent point
        x0 = zeros(1,dim); x0(tp_dim) = -(r_small + 2*rand*threshold);
        assertTrue(testCase, costfun(x0) <= threshold, 'Blah! Smth wrong w/ x0 or the cost function.');
        x0_ca{dim_idx} = x0;
        % bounds
        bmax_ca{dim_idx} = 10*ones(1,dim);
        bmin_ca{dim_idx} = -10*ones(1,dim);
    end

    for i=1:size(nfevalProportionsArray,1)
        nfevalProportions = nfevalProportionsArray(i,:);
        name = sprintf('%s_%s', testName, regexprep(int2str(nfevalProportions),'\s',''));
        integrateBulk(testCase, name,...
            dim_vec, nfeval_vec, nfevalProportions, nrep, parallelize,...
            costfun, threshold, x0_ca, bmin_ca, bmax_ca, {},...
            vol_vec, volRelErrTol, saveMat);
    end
end

function testIntegrateWurst(testCase)
%TESTINTEGRATEWURST Piecewise random cylinder & spheres at the joints.
%

    testName = 'Wurst';

    % bulk integration setup
    nfeval_vec = testCase.TestData.bulk.nfeval_vec;
    dim_vec = testCase.TestData.bulk.dim_vec;
    volRelErrTol = testCase.TestData.bulk.volRelErrTol;
    nrep = testCase.TestData.bulk.nrep;
    parallelize = testCase.TestData.bulk.parallelize;
    nfevalProportionsArray = testCase.TestData.bulk.nfevalProportionsArray;
    saveMat = testCase.TestData.bulk.saveMat;

    % setup per dimension, indlucing cost function and threshold
    for i=numel(dim_vec):-1:1
        dim = dim_vec(i);
        [ x0_ca{i}, costfun_ca{i}, threshold_vec(i), bmin_ca{i}, bmax_ca{i}, vol_vec(i) ] = ...
            hstest.wurst.initRandWurst(dim);
    end

    for i=1:size(nfevalProportionsArray,1)
        nfevalProportions = nfevalProportionsArray(i,:);
        name = sprintf('%s_%s', testName, regexprep(int2str(nfevalProportions),'\s',''));
        integrateBulk(testCase, name,...
            dim_vec, nfeval_vec, nfevalProportions, nrep, parallelize,...
            costfun_ca, threshold_vec, x0_ca, bmin_ca, bmax_ca, {},...
            vol_vec, volRelErrTol, saveMat);
    end
end
