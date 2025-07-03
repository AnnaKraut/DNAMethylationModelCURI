function [out1, out2] = functionfinal2(varargin)
%FUNCTIONFINAL2 Same as functionfinal, but cost is non-zero iff indicator is
% non-zero.
    [out1, out2]=functionfinal(varargin{:});
    if isnumeric(varargin{1})
        if ~out1 % indicator == 0
            out2 = out1; % cost := 0
        end
    end
end
