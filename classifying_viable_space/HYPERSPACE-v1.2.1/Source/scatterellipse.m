function h = scatterellipse(V0, V, cost, E, c, titleStr)
%SCATTERELLIPSE Plot 2-D scatter plot of viable points, their cost and the
% minimal viable enclosing ellipsoids found.
%
% To plot ellipses, install:
%     https://www.mathworks.com/matlabcentral/fileexchange/289-ellipse-m
%
    dim = size(c,1);
    if isempty(V0)
        V0dim = dim;
    else
        V0dim = size(V0,2);
    end
    assert(~any(diff([V0dim size(V,2) size(E,1) dim])), 'Points dimensions mismatch.');
    if (dim ~= 2), error('scatterellipse works only in 2-D'); end
    assert(dim == size(E,2), 'Number of ellipsoids axes mismatch.');
    nelip = size(E,3);
    assert(nelip == size(c,3), 'Number of ellipsoids mismatch.');
    assert(numel(cost) == size(V,1), 'Number of samples and costs mismatch.');

    figure;
    ptsize = 100;
    if ~isempty(V0)
        scatter(V0(:,1),V0(:,2),ptsize,'k.');
        hold on;
    end
    [sVcost, iV] = sort(cost,'descend'); % to plot low value points last
    scatter(V(iV,1),V(iV,2),ptsize,sVcost,'.');
    %axis('equal');
    colorbar('peer',gca); % ,'location','eastoutside'

    if exist('ellipse', 'file')
        for i=1:nelip
            [D, W] = principaxesellip(E(:,:,i),dim);
            assert(sum(W(1,:) .* W(2,:)) < 1e12, 'Principle axes of ellipse are not prependicular.');
            ellipse(D(1),D(2),atan2(W(1,2),W(1,1)),c(1,1,i),c(2,1,i),'r');
        end
    else
        warning('Ellipses not plotted; install: https://www.mathworks.com/matlabcentral/fileexchange/289-ellipse-m .');
    end
    title(titleStr);
    hold off;
    h = gcf;
end
