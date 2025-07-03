%% Load
tn = 'Donut';
dirnames = dir(['test_' tn '*.mat']);

n = numel(dirnames);
ResultsPerSampleProportions = cell(n,1);
for isampprop=1:n
    fn = dirnames(isampprop).name;
    ResultsPerSampleProportions{isampprop} = load(fn);
end

%% Plot
% Create plots per proportions:
for isampprop=1:n
    R = ResultsPerSampleProportions{isampprop};

    % dimensions
    dim_vec = R.dim_vec;
    ndim = numel(dim_vec);
    % true volumes
    vol_true = R.vol_vec;

 
    figure();

    for idim=1:ndim
        % extract running times, number of feval and computed volumes
        r_times   = squeeze(R.nsecArray(idim,:,:));
        time_min = min(min(r_times))/1.25;
        time_max = max(max(r_times))*1.25;
        r_fevals  = squeeze(R.nfevalArray(idim,:,:));
        nfeval_min = min(min(r_fevals))/1.25;
        nfeval_max = max(max(r_fevals))*1.25;
        r_outv = squeeze(R.OutVArray(idim,:,:));
        r_relvols = reshape([r_outv.vol], size(r_outv))/vol_true(idim);
        relvol_min = min(min(r_relvols))/1.25;
        relvol_max = max(max(r_relvols))*1.25;

        dim = dim_vec(idim);
        subplot(ndim,3,(idim-1)*3+1); hold on;
        plot(r_fevals', r_relvols', '+'); grid on;
        plot( get(gca,'XLim'), ones(1,2),'--k');
        axis([nfeval_min nfeval_max relvol_min relvol_max]); xlabel('fevals'); ylabel('Rel. vol');
        title(sprintf('%s (%dd)',tn,dim));

        subplot(ndim,3,(idim-1)*3+2); hold on;
        plot(r_times', r_relvols', '+'); grid on;
        plot( get(gca,'XLim'), ones(1,2),'--k');
        axis([time_min time_max relvol_min relvol_max]); xlabel('time (s)'); ylabel('Rel. vol');
        title(sprintf('%s (%dd)',tn,dim));

        subplot(ndim,3,(idim-1)*3+3);  hold on;
        plot(r_fevals', r_times', '+'); grid on;
        axis([0 Inf 0 Inf]); xlabel('fevals'); ylabel('time (s)');
        title(sprintf('%s (%dd)',tn,dim));
    end
   
    axes('Position',[0 0 1 1],'Visible','off');
    text(0.1,0.975, sprintf('Sample proportions (OEAMC+MEBS+Volint): %.3g%%+%.3g%%+%.3g%%',100*R.nfevalProportions));

end
