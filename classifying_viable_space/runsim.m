function [methylated, unmethylated, middle, sorta] =runsim(params)
props = pyrunfile("run_sim.py","output",r_hm=params(1),r_hm_h=params(2),r_uh=params(3),r_uh_h=params(4),r_mh=params(5),r_mh_h=params(6),r_hu=params(7),r_hu_h=params(8),r_cell_div=1);

props = double(props)

methylated = props(1);
unmethylated = props(2);
middle = props(3);
sorta = props(4);

end