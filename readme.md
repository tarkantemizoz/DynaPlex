![Dynaplex logo](docs/source/assets/images/logo.png)

This repository replicates the results presented in the papers:
- *[Deep Controlled Learning for Inventory Control](https://www.sciencedirect.com/science/article/pii/S0377221725000463)*
- *[Zero-shot Generalization in Inventory Management: Train, then Estimate and Decide](https://arxiv.org/abs/2411.00515)*

For the original repository, including the latest updates and documentation, please visit [DynaPlex on GitHub](https://github.com/DynaPlex/DynaPlex).
Please do not hesitate to contact the first author when replicating the results or if you have any questions. Make sure to include IO_DynaPlex folder to replicate all the results. 

For the paper titled "Deep Controlled Learning for Inventory Control", please see folder **`DeepControlledLearning/`** for the weights of the neural networks for each inventory setting, and  **`src/lib/models/models/`** folder for the construction of MDP-EI's of the inventory problems.
---

For the paper titled "Zero-shot generalization in Inventory Management: Train, then Estimate and Decide", please see folder **`GC-LSN_weights/`** for the weights of the generally capable lost sales network GC-LSN, and  **`src/lib/models/models/Zero_Shot_Lost_Sales_Inventory_Control`** for the construction of the Super-Markov Decision Process of lost sales inventory control problem.
---

## High-level overview of folder structure

- **`LICENSES/`**: Contains the licenses to used libraries and packages.
- **`DeepControlledLearning/`**: Contains the test results and policy weights for the inventory problems presented in Deep Controlled Learning in Inventory Control paper.
- **`GC-LSN_weights/`**: Contains the weights for the generally capable agent  GC-LSN  for lost sales inventory control.
- **`IO_DynaPlex/`**: Folder where all networks weights are written and can be inferred to.
- **`bash/`**: Contains the files used for running on a Linux HPC.
- **`cmake/`**: Contains support functionality for building with CMake. 
- **`docs/`**: Contains the documentation.
- **`python/`**: Contains example python scripts, that can be used after building the python bindings.
- **`src/`**: Contains the main code base
  - **`executables/`**: Contains all executables you can run (you can add additional executables yourself here, that use the library).
  - **`extern/`**: Contains all external libraries used (e.g., googletest).
  - **`lib/`**: Contains all algorithms and all MDP models, you can implement your MDP in src/lib/models/models.
  - **`tests/`**: Contains all code for unit testing (supported by googletest).

---

## Building and running executables

The repository ships with helper scripts under `bash/` that build a target with CMake and then run it, logging stdout/stderr to `bash/logs/<exe>_<timestamp>.{out,err}`. Build presets (`MacRel`, `LinRel`, `LinMPI`) come from `CMakePresets.json` / `CMakeUserPresets.json`; configure them once before the first run (e.g. `cmake --preset LinRel`).

### macOS (local)

```bash
cd bash
./mac.sh <executable> [args...]                       # builds + runs with MacRel
PRESET=MacDeb ./mac.sh <executable>                   # override the preset
```

### Linux HPC (Snellius example)

The cluster scripts are full sbatch jobs — they load the required modules, build the target, then `srun` it. You do **not** need to `source loadmodules.sh` first; everything is self-contained.

```bash
cd bash

# single-node job (preset = LinRel, 1 node, 10h, 192 cpus, 336G)
sbatch linux.job <executable> [args...]

# multi-node MPI job (preset = LinMPI, 5 nodes, 20h)
sbatch linux_mpi.job <executable> [args...]

# override any sbatch directive at submit time
sbatch --time=4:00:00 --nodes=2 linux_mpi.job <executable>
```

Both `.job` files resolve their repo location from the submitted script path, so they work correctly even if you keep multiple clones of the repo on the cluster. Module versions in `linux.job` / `linux_mpi.job` are pinned to the Snellius 2023 stack (CMake 3.26.3, OpenMPI 4.1.5); adjust them if your cluster exposes different module names.

If anything is unclear or you hit cluster-specific issues, feel free to contact the author of this repository for guidance.

---