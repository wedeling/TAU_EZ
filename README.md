# ECCOMAS-UNCECOMP 2019

This branch of the TAU_EZ is meant to reproduce the results of our UNCECOMP19 conference paper:

W. Edeling, D. Crommelin, "Reduced model-error source terms for fluid flow", UNCECOMP 19 Conference, Crete, June 24-26, 2019.

## Abstract
It is well known that the wide range of spatial and temporal scales present in geophysical flow problems represents a (currently) insurmountable computational bottleneck, which must be circumvented by a coarse-graining procedure. The effect of the unresolved eddy field enters the coarse-grained equations as a unclosed forcing term, denoted as the 'eddy forcing'. Traditionally, the system is closed by approximate deterministic closure models, i.e. so-called parameterizations. Instead of creating a deterministic parameterization, some recent efforts have focused on creating a stochastic, data-driven surrogate model for the eddy forcing from a (limited) set of reference data, with the goal of accurately capturing the long-term flow statistics. Since the eddy forcing is a dynamically evolving field, a surrogate should be able to mimic the complex spatial patterns displayed by the eddy forcing. Rather than creating such a (fully data-driven) surrogate, we propose to precede the surrogate construction step with a procedure that replaces the eddy forcing with a new model-error source term which: i) is tailor-made to capture spatially-integrated statistics of interest, ii) strikes a balance between physical insight and data assimilation, and iii) significantly reduces the amount of training data. Instead of creating a surrogate for an evolving field, we now only require a surrogate model for one scalar time series per statistical quantity-of-interest. Our current surrogate modelling approach builds on a resampling strategy, where we create a probability density function of the reduced training data that is conditional on (time-lagged) resolved-scale variables. We will derive the model-error source terms, and construct the reduced surrogate using an ocean model of two-dimensional turbulence in a doubly periodic square domain.

## Funding
This research is funded by the Netherlands Organization for Scientific Research (NWO) through the Vidi project "Stochastic
models for unresolved scales in geophysical flows", and from the European Union Horizon 2020 research and innovation programme under grant agreement \#800925 ([VECMA](https://www.vecma.eu/) project). 

## Reproduction of main results

### Dependencies
+ Python 2
+ Numpy
+ Scipy
+ Matplotlib
+ [h5py](https://github.com/h5py/h5py)
+ (Optional) [drawnow](https://github.com/stsievert/python-drawnow)

In order to reproduce the probability density functions of the reduced training data (Section 3.2) run:

+ *python tau_ez_ocean.py ./inputs/training.json*

This generates the training data in the ./samples directory, which can be visualized with:

+ *python plot_EZ_stats.py ./inputs/training.json*

Likewise, the main results of Section 4 can be reproduced with:

+ *python tau_ez_ocean.py ./inputs/T2.json*
+ *python plot_EZ_stats.py ./inputs/T2.json*

and

+ *python tau_ez_ocean.py ./inputs/T3.json*
+ *python plot_EZ_stats.py ./inputs/T3.json*
