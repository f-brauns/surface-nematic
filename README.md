This repository cotains the simulation code for the PNAS article 

## Patterning of morphogenetic anisotropy fields

by Zihang Wang, M. Cristina Marchetti, and Fridtjof Brauns.
The mathematical background is described in the Methods section of the paper.

Simulations of nematic textures on curved, formulated using the Q-tensor formalism, were performed both in Python (using FEniCS) and in COMSOL Multiphyisics v6.0. The form language in FEniCS allows a very elegant and transparent implementation of the mathematical expressions for the covariant derivative of the Q-tensor on the surface and the out-of-plane penalty of the Q-tensor. Setting up the simulations in COMSOL is slightly more involved. To implement the mathematical expressions we use "Variables" in the "Definitions" node in COMSOL's structure tree. The expressions are assembled using a Mathematice script, which exports them in a format that can be loaded in COMSOL. COMSOL outperforms FEniCS by several orders of magnitude. In addition, it allows one to perform simulations on deforming geometries with automatic remeshing.
