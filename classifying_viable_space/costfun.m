function [cost] =costfun(params)
X = 0;
props = pyrunfile("run_sim.py","output",r_hm=params(1),r_hm_h=params(2),r_uh=params(3),r_uh_h=params(4),r_mh=params(5),r_mh_h=params(6),r_hu=params(7),r_hu_h=params(8),r_cell_div=1);
props = double(props); % gotta make this useable in MatLab; before it is a weird python data type
% if we spend 80+% in the middle, it is not bistable --> cost of infinity
if props(3) > .8
    cost = inf;
% otherwise the cost is the ratio of the difference between the two regions to their sum
else
    cost = abs(props(1) - props(2) + X) / (props(1) + props(2));
end
end