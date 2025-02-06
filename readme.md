![Dynaplex logo](docs/source/assets/images/logo.png)

This repository replicates the results presented in the papers:
- *[Deep Controlled Learning for Inventory Control](https://www.sciencedirect.com/science/article/pii/S0377221725000463)*
- *[Zero-shot Generalization in Inventory Management: Train, then Estimate and Decide](https://arxiv.org/abs/2411.00515)*

For the original repository, including the latest updates and documentation, please visit [DynaPlex on GitHub](https://github.com/DynaPlex/DynaPlex).

For the paper titled "Deep Controlled Learning for Inventory Control", please see folder **`DeepControlledLearning/`** for the weights of the neural networks for each inventory setting, and  **`src/lib/models/models/`** folder for the construction of MDP-EI's of the inventory problems.
---

For the paper titled "Zero-shot generalization in Inventory Management: Train, then Estimate and Decide", please see folder **`GC-LSN_weights/`** for the weights of the generally capable lost sales network GC-LSN, and  **`src/lib/models/models/Zero_Shot_Lost_Sales_Inventory_Control`** for the construction of the Super-Markov Decision Process of lost sales inventory control problem.
---

## High-level overview of folder structure

- **`LICENSES/`**: Contains the licenses to used libraries and packages.
- **`DeepControlledLearning/`**: Contains the test results and policy weights for the inventory problems presented in Deep Controlled Learning in Inventory Control paper.
- **`GC-LSN_weights/`**: Contains the weights for the generally capable agent  GC-LSN  for lost sales inventory control.
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