#include <iostream>
#ifdef DP_MPI_AVAILABLE
#include <mpi.h>
#endif
#include "dynaplex/torchavailability.h"
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/trainedpolicyprovider.h"

namespace DynaPlex {


    // Implementing the Singleton pattern for DynaPlexProvider
    DynaPlexProvider& DynaPlexProvider::Get() {
        static DynaPlexProvider instance;  // Guaranteed to be lazy initialized and destroyed correctly
        return instance;
    }
    std::string DynaPlexProvider::FilePath(const std::vector<std::string>& subdirs, const std::string& filename)
    {
        return m_systemInfo.filepath(subdirs, filename);
    }
    void DynaPlexProvider::SavePolicy(DynaPlex::Policy policy, std::string file_path_without_extension) {
        TrainedPolicyProvider::SavePolicy(policy, file_path_without_extension);        
    }


    DynaPlex::Policy DynaPlexProvider::LoadPolicy(DynaPlex::MDP mdp, std::string file_path_without_extension) {
        return TrainedPolicyProvider::LoadPolicy(mdp, file_path_without_extension);
    }

    DynaPlex::Algorithms::ExactSolver DynaPlexProvider::GetExactSolver(DynaPlex::MDP mdp, const VarGroup& config)
    {
        return DynaPlex::Algorithms::ExactSolver{ this->System(),mdp,config };
    }

    DynaPlex::Algorithms::DCL DynaPlexProvider::GetDCL(DynaPlex::MDP mdp, DynaPlex::Policy policy, const VarGroup& config)
    {
        return DynaPlex::Algorithms::DCL{ this->System(),mdp, policy,config };
    }

   
    void DynaPlexProvider::SetIORootDirectory(std::string path) {
        m_systemInfo.SetIOLocation(path, "IO_DynaPlex");
    }

    void DynaPlexProvider::AddBarrier()
    {
#ifdef DP_MPI_AVAILABLE
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    DynaPlexProvider::DynaPlexProvider() {
        // If MPI is available, initialize it and fetch world details
#ifdef DP_MPI_AVAILABLE
        int mpi_initialized;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized) {
            MPI_Init(nullptr, nullptr);
        }

        int world_rank;
        int world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
        int world_rank = 0;  // Default values
        int world_size = 1;
#endif
        bool torchavailable = DynaPlex::TorchAvailability::TorchAvailable();
      
        m_systemInfo = DynaPlex::System(torchavailable,world_rank, world_size,
           /*callback function: */ []() {DynaPlexProvider::Get().AddBarrier(); }
            );
        std::string defined_root_dir = "";
#ifdef DYNAPLEX_IO_ROOT_DIR
        defined_root_dir = DYNAPLEX_IO_ROOT_DIR;
#endif
        if (!defined_root_dir.empty())
        { 
            try
            {
                m_systemInfo.SetIOLocation(defined_root_dir, "IO_DynaPlex");
            }
            catch (const DynaPlex::Error& e)
            {
                // Print the specific error message from the caught exception.
                std::cerr << "Error: " << e.what() << std::endl;

                // Construct an informative message.
                std::string informativeMsg = "The root directory provided as a compiler definition DYNAPLEX_IO_ROOT_DIR (e.g. from CMakeUserPresets.json or during compilation) "
                    "(" + defined_root_dir + ") is likely not an existing directory. "
                    "Please verify the provided path and recompile.";

                // Throw a new exception or re-throw the original with an updated message.
				throw DynaPlex::Error(informativeMsg);
			}
		}
#ifdef DP_MPI_AVAILABLE
		std::cout << "DynaPlex: Hardware Threads: " << m_systemInfo.HardwareThreads()
			<< ", MPI: Yes, Rank: " << world_rank
            << "/" << world_size << std::endl;
#else
        std::cout << "DynaPlex: Hardware Threads: " << m_systemInfo.HardwareThreads()
            << ", MPI: No, " << std::endl;
#endif
        AddBarrier();
        if (torchavailable)
            m_systemInfo << DynaPlex::TorchAvailability::TorchVersion() << std::endl;
        else
            m_systemInfo << "Torch not available" << std::endl;
        AddBarrier();
        // Register all the MDPs upon startup.
        Models::RegistrationManager::RegisterAll(m_registry);
    }


    // Destructor
    DynaPlexProvider::~DynaPlexProvider() {
     
        System() << "DynaPlex: Finalizing. Time Elapsed: " << System().Elapsed() << std::endl;
        // If MPI is available, finalize it
#ifdef DP_MPI_AVAILABLE
        int mpi_finalized;
        MPI_Finalized(&mpi_finalized);
        if (!mpi_finalized) {
            MPI_Finalize();
        }
#endif
    }

    MDP DynaPlexProvider::GetMDP(const VarGroup& config) {
        return m_registry.GetMDP(config);
    }

    VarGroup DynaPlexProvider::ListMDPs() {
        return m_registry.ListMDPs();
    }

    const System& DynaPlexProvider::System() {
        if (!m_systemInfo.HasIODirectory())
        {
            throw DynaPlex::Error("You used functionality that may require a valid input/output directory, but none is available. Ensure that DynaPlexProvider::Get().SetIORootDirectory is called before using this functionality. Alternatively, provide compiler-defined macro DYNAPLEX_IO_ROOT_DIR representing a valid root directory.");
        }
        return m_systemInfo;
    }

    DynaPlex::Utilities::Demonstrator DynaPlexProvider::GetDemonstrator(const VarGroup& config)
    {
        return DynaPlex::Utilities::Demonstrator(m_systemInfo, config);
    }

  
    DynaPlex::Utilities::PolicyComparer DynaPlexProvider::GetPolicyComparer(DynaPlex::MDP mdp, const VarGroup& config)
    {
        return DynaPlex::Utilities::PolicyComparer(m_systemInfo,mdp, config);
    }

}  // namespace DynaPlex
