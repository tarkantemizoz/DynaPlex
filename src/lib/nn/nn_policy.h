#pragma once
#include "dynaplex/mdp.h"
#include "dynaplex/policy.h"
#include "neuralnetworkprovider.h"


// Forward declarations
namespace torch {
    namespace nn {
        class AnyModule;
    }
}


namespace DynaPlex {



    class NN_Policy : public PolicyInterface {
    public:
        enum class NetworkForwardType {
            Tensor,
            TensorDict,
            TensorDictMask
        };

        NetworkForwardType fw_type = NetworkForwardType::Tensor;

        DynaPlex::MDP mdp;
#if DP_TORCH_AVAILABLE
        std::unique_ptr<torch::nn::AnyModule> neural_network;
#endif
        DynaPlex::VarGroup policy_config;
        NN_Policy(DynaPlex::MDP mdp);

        std::string TypeIdentifier() const override;

        const DynaPlex::VarGroup& GetConfig() const override;

        void SetAction(std::span<Trajectory> trajectories) const override;

        std::vector<int64_t> GetPromisingActions(const DynaPlex::dp_State& dp_state, int64_t num_actions) const override;

    };

}  // namespace DynaPlex

