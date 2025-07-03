function [out1, out2] = functionfinal(varargin)
%FUNCTIONFINAL ...
%
% Syntax:
%     functionfinal('S', threshold, n, dim, fun, fargs)
%         setup functionfinal
%     [cuenta, viable] = functionfinal('G')
%         get persistent data
%     [indicator, cost] = functionfinal(p)
%         evaluate fun(p, fargs{:}) and check threshold
%
    persistent p;

    v1 = varargin{1};
    if isnumeric(v1)
        % assert(~isempty(p));
        p.cuenta=p.cuenta+1;
        try
            cost=feval(p.fun,v1,p.fargs{:});
        catch ME
            warning('FUNCTIONFINAL:CostEvaluationError','Error while evaluating cost function at %s:\n\t%s: %s\n', mat2str(v1), ME.identifier, ME.message);
            cost = Inf;
        end

        isviable = (cost <= p.threshold);
        if isviable
            U=nviablepts(p.viable);
            p.viable(U+1,1:(p.dim+1))=[v1 cost];
        end
        indicator = double(isviable);
        out1 = indicator;
        out2 = cost;
    else
        switch v1
        case '-s' % setup persistent data
            p.cuenta = 0;
            p.threshold = varargin{2};
            n = varargin{3};
            p.dim = varargin{4};
            p.viable=initviable(n,p.dim);
            p.fun = varargin{5};
            p.fargs = varargin{6};
        case '-g' % get persistent data
            0;
        otherwise % evaluate fun using persistent data
            error('FUNCTIONFINAL:InvalidSwitch','Invalid switch: ''%s''.',v1);
        end
        out1=p.viable;
        out2=p.cuenta;
    end

end
