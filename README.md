# ml_research

Research internship with the Grayson Group, University of Bath, 20 July – 11 September 2026.
Active and still moving.

The question is whether a large language model can usefully warm-start hyperparameter
optimisation for chemistry ML - using an LLM's prior over plausible configurations to seed a
Bayesian optimiser, rather than starting it cold. So far I've built an end-to-end support
vector regression pipeline predicting DFT reaction barriers, deployed on a Linux compute
cluster, and benchmarked LLM warm starts against grid search, random search and standard
Bayesian optimisation. Warm starts converged faster and reached a test error of 0.98 kcal/mol,
below the 1.0 target for the project. The next phase, running through to September, extends
the method to a graph neural network for aqueous solubility prediction.

Worth saying plainly that this is a learning project as much as a research one. I came to it
from a chemistry background rather than a CS one, so a fair amount of what's here is me
working out the tooling as I go, and the structure will likely change again before September.

**Stack:** Python · scikit-learn · Optuna · PyTorch · PyTorch Geometric · RDKit · pandas · NumPy · matplotlib

Jacob Mlang'a — MSci Chemistry with Molecular Physics, Imperial College London
