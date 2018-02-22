# Minimal Distributed Heat Equation
Minimal Distributed Heat Equation model based on deal.II tutorials - step-17 and step-18.

### Capabilities
* Solution of transient heat equation in 2D and 3D
* Neuman boundary conditions (convection)
* Dirichlet boundary conditions (constant temperature)
### Requirements
* Modern C++ compiler (works with gcc-5.3.1)
* cmake (works with cmake-3.7.2)
* Deal.II built with MPI and PETSc. Tested with Deal.II 8.5.1, PETSc 3.8.0, OpenMPI 1.10.3 (from RHEL7 repository)

### Build and run
```mkdir Debug
cd Debug
cmake ../ -DCMAKE_BUILD_TYPE=Release
mpirun -n 8 dhe