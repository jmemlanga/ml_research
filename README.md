# ml_research

Work in progress. Research internship with the Grayson Group, University of Bath,
20 July – 11 September 2026.

The aim is to test whether a large language model can be used to warm-start hyperparameter
optimisation for chemistry ML models — using an LLM's prior over sensible configurations to
seed a Bayesian optimiser, rather than starting it cold. Right now I'm still at the front end
of that: sourcing and cleaning datasets, building a support vector regression pipeline for DFT
reaction barriers, and standing up grid search, random search and Optuna's `GPSampler` as
baselines to measure anything against. The warm-start comparison itself is the goal for the
back half of the internship, alongside an extension to a graph neural network for aqueous
solubility prediction.

Fair warning that this is exploratory rather than polished. I came into it from a chemistry
background with no formal ML training, so a lot of what's here is me learning the tooling as I
go, and the structure will likely change more than once before September.

**Stack:** Python · scikit-learn · Optuna · PyTorch · PyTorch Geometric · RDKit · pandas · NumPy · matplotlib

Jacob Mlang'a — MSci Chemistry with Molecular Physics, Imperial College London
