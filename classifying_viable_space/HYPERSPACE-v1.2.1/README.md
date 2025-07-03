Intro
=====

This MATLAB package implements the parameter space exploration method described by Zamora-Sillero, E. et al. [[1]].


Installation
============

Download the latest release of the HYPERSPACE toolbox [[2]]. Add `Source` folder (with sub-folders) to your MATLAB path (w/o examples and tests folders).


Quickstart
==========

See the tutorial PDF in `Docs/` (note: it was written for ver. 1.0). For the complete up to date workflow example have a look at `Tests/AllTest>integrate` function. 

Once you get an idea of what is the intended way of usage of the package, it is highly recommended to test your setup by running the tests, in particular:

```matlab

    runtests('VolintTest')
    runtests('AllTest')
```

Authors
=======

Initial (ver. 1.0):
    
    Elías Zamora-Sillero <e.zamora@bioc.uzh.ch>

Current (ver >1.0):
    
    Mikołaj Rybiński <mikolaj.rybinski@bsse.ethz.ch>
    Thomas Liphardt <thomas.liphardt@bsse.ethz.ch>

References
==========

[1]: http://www.biomedcentral.com/1752-0509/5/142
1. [HYPERSPACE paper @ BioMed Central][1]
[2]: https://gitlab.com/csb.ethz/HYPERSPACE/tags
2. [HYPERSPACE toolbox releases @ GitLab.com][2]
