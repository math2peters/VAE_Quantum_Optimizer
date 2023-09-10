# Introduction
  In neutral atom experiments, many experimental system parameters often can not be analytically modeled well enough to simulate the resulting atom dynamics and predict optimal control sequences (for example, the optimal pulse profile to prepare an atom in state |1> given laser noise, drifting parameters, etc.). In an attempt to solve this problem, this repository implements Variational AutoEncoder (VAE) Quantum Optimizer for use in real-time quantum experiments involving neutral atoms. It draws from the latent space of a VAE to attempt to optimize a discretely-sampled series of parameters to achieve a desired outcome. 
 
  There are many difficulties to overcome in attempting to optimize neutral atom experiments:
  1. Experiments are subject to drifting parameters over the timescale of ~1 hour due to factors such as sensitive laser alignment
  2. The cycle rate is often slow (~1 data point/second), leading to small sample sizes
  3. Small sample sizes then lead to overfitting and mode collapse
  4. Measurement during the sequence often affects the state of the system, which leads to difficulty in extracting information from a trial (except at the end of the sequence)
  5. Parameters spaces are often very sparse, with only a tiny subset of the space providing enough signal to get meaningful information. For example, most combinations of parameter choices will eject atoms from the system

# VAE Quantum Optimizer 
 This repository is a framework for solving this problem of optimizing neutral atom experiments. The basic idea is:
 1. Encode into a VAE latent space what a "reasonable" function for optimizing a sequence may look like. This teaches the VAE how to represent functions such as Gaussians, Sines, Sincs, Lines, Exponentials, and superpositions of these in a lower dimensional space that can be more easily searched. Specificially, these functions are sampled at 64 points and encoded into a latent space of dimension 8. If there is prior knowledge about the problem, the VAE could be fine-tuned to represent certain function classes such as wavelets. 
 2. 

 2. Choose many random but reasonable superpositions of Gaussians, Sines, etc. for f(t). Numerically calculate the final population in state |1> after evolving the system using QuTiP. This forms the base of the dataset that will be used to train the VAE

 Mode collapse prevent
 Teaches the architecture to learn reasonable functions

 Start with the Hamiltonian H = -$$\sigma _z$$/2 + f(t) $$\sigma _x$$ that evolves under the strong collapse operator $$\Gamma$$ |0><1|. If $$\Gamma$$ is large enough,
