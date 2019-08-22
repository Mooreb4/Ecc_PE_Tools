# Ecc_PE_Tools
Code for PE using the TaylorF2e model: Fisher, Likelihood, Simulated Annealing (ish), MCMC

Fisher uses a numerically stable scheme to calculate numerical derivatives of the model (an eccentric/many harmonic generalization of arXiv:1007.4820)
The fisher code also checks for the condition number of the fisher and adjusts such that inversion does not lead to numerical inaccuracy

Likelihood is maximized over the extrinsic parameters of the eccentric model (t_c, l_c, lambda_c)

The sys_ff.cpp is meant to find the maximum likelihood parameters very accurately using a very large SNR injection and many parallel tempering chains

MCMC code uses a proposal distribution which is mix of fisher jumps, draws from prior, differential evolution, and parallel tempering.

