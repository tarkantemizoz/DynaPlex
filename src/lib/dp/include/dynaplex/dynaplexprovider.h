#pragma once
#include "dynaplex/vargroup.h"
#include "dynaplex/models/registrationmanager.h"
#include "dynaplex/registry.h"
#include "dynaplex/system.h"
#include "dynaplex/demonstrator.h"
#include "dynaplex/policycomparer.h"
#include "dynaplex/dcl.h"
#include "dynaplex/exactsolver.h"
namespace DynaPlex {
    class DynaPlexProvider {
        
    public:
        /// provides access the single instance of the class
        static DynaPlexProvider& Get();

        /**
         * sets the root directory where a IO_DynaPlex subdirectory will be created, where all 
         * input and output from dynaplex will be nested. 
         */ 
        void SetIORootDirectory(std::string path);


        /// gets an MDP based on the vargroup 
        MDP GetMDP(const VarGroup& config);
        /// lists the MDPs available. 
        VarGroup ListMDPs();


        const DynaPlex::System& System();


        std::string FilePath(const std::vector<std::string>& subdirs, const std::string& filename);

        void SavePolicy(DynaPlex::Policy policy, std::string file_path_without_extension);

        DynaPlex::Policy LoadPolicy(DynaPlex::MDP mdp, std::string file_path_without_extension);
        


        /**
         * @brief gets an instance of the exact solver
         * @param mdp model
         * @param configuration object for the exact solver.
         * @return the configured exact solver.
        */
        DynaPlex::Algorithms::ExactSolver GetExactSolver(DynaPlex::MDP mdp, const VarGroup& config = VarGroup{});


        /**
         * @brief gets an instance of the dcl solver
         * @param mdp model
         * @param policy used to kickstart/initiate the solution process. random policy is used of no policy is provided.
         * @param configuration object for dcl algorithm. May provide H (default 40/256 for finite/infinite horizon MDP), M (default 1000), N (default 5000), L (only for infinite horizon; default 100). May also provide num_gens (default 1), nn_training settings, and nn_architecture settings.
         * @return the configured dcl instance.
        */
        DynaPlex::Algorithms::DCL GetDCL(DynaPlex::MDP mdp, DynaPlex::Policy policy = nullptr, const VarGroup& config = VarGroup{});

     

        /**
         * Config may include max_period_count (default:3)
         * it may also include rng_seed (default:0).
         */
        DynaPlex::Utilities::Demonstrator GetDemonstrator(const VarGroup& config = VarGroup{});

        /**
         * Gets a policy evaluator for a specific mdp. A algorithm config may also be provided.
         * Config may include number_of_trajectories (default:4096 for infinite horizon mdps; 16384 for finite horizon mdps).
         * If mdp is infinite horizon, undiscounted: config may include warmup_periods (default: 128), periods_per_trajectory (default: 1024).
         * If mdp is infinite horizon, discounted: config may include periods_per_trajectory (default: 1024).
         * If mdp is finite horizon: config may include max_periods_until_error (default: 16384), this is the maximum number of steps in a trajectory until mdp is expected to terminate by reaching final state.
         * Config may also include rng_seed (default 0).
         */
        DynaPlex::Utilities::PolicyComparer GetPolicyComparer(DynaPlex::MDP mdp, const VarGroup& config = VarGroup{});


    private:
        void AddBarrier();
        DynaPlexProvider(); 
        ~DynaPlexProvider();
        // Delete the copy and assignment constructors to ensure singleton behavior
        DynaPlexProvider(const DynaPlexProvider&) = delete;
        DynaPlexProvider& operator=(const DynaPlexProvider&) = delete;

        Registry m_registry;          // private instance of Registry
        DynaPlex::System m_systemInfo;      // private instance of System
    };

}  // namespace DynaPlex
