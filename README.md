# VAE Quantum Optimizer
 VAE quantum optimizer is an architecture for use in quantum experiments involving neutral atoms. 

 In neutral atom experiments, the cycle rate is often slow (~1 data point/second) and experiments are additionally subject to drifting parameters over the timescale of ~1 hour due to things such as sensitive laser alignment. Additionally, it is rare to be able to measure the system except at the end of the optimization run because measurement often destroys or drastically affects the system you're trying to measure. Finally, the most difficult to address problem is that the parameter spaces for these experiments are extremely sparse, and changing one part of an experimental sequence can often lead to no measureable signal (for example, if all the atoms get ejected by a measurement). Together, these problems preclude the application of most standard algorithms for optimization.
 
 This repository is a framework for solving this problem of optimizing neutral atom experiments. The basic idea is:
 1. Start with the Hamiltonian H = -$$\sigma _z$$/2 + f(t) $$\sigma _x$$ that evolves under the strong collapse operator $$\Gamma$$ |0><1|. If $$\Gamma$$ is large enough, 
 2. Choose many random but reasonable superpositions of Gaussians, Sines, etc. for f(t). Numerically calculate the final population in state |1> after evolving the system using QuTiP. This forms the base of the dataset that will be used to train the VAE
