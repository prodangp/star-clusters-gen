# Generating synthetic star clusters using Gaussian processes

The birthplaces of stars are complex places, where turbulent interstellar gas collapses and fragments into star-forming cores, giving rise to non-trivial substructure. While the formation process can be modelled with hydrodynamical simulations, these are quite expensive in terms of computational resources. Moreover, primordial star clusters that are still embedded in their parent gas cloud are hard to constrain observationally. In this context, most efforts aimed at simulating the dynamical evolution of star clusters assume simplified initial conditions, such as truncated Maxwellian models.
We aim to improve on this state-of-the-art by introducing a set of tools to generate realistic initial conditions for star clusters by training an appropriate class of machine learning models on a limited set of hydrodynamical simulations. In particular, we will exploit a new approach based on Gaussian process (GP) models, which have the advantage of differentiability and of being more tractable, allowing for seamless inclusion in a downstream machine learning pipeline for e.g. inference purposes. The proposed learning framework is a two-step process including the model training and the sampling of new stellar clusters based on the inference results. We investigate different sampling approaches in order to find samplers that are able to generate realistic realizations.



### Features 
<ul>
    <li>Generates synthetic star clusters with customizable size and density distributions
    <li>Supports Gaussian process regression for generating new density maps from observed data
    <li>Several basic sampling methods: MCMC with Metropolis, APES, Rejection, and others.
    <li>A physical-informed sampling method (EMCMC)
</ul>

### How To Use

