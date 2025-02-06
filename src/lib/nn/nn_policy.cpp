#include "nn_policy.h"
#include "dynaplex/system.h"
#if DP_TORCH_AVAILABLE
#include <torch/torch.h>
#endif
namespace DynaPlex {

	NN_Policy::NN_Policy(DynaPlex::MDP mdp)
		: mdp(mdp) {

	}

	std::string NN_Policy::TypeIdentifier() const {
		return "NN_Policy";
	}

	const DynaPlex::VarGroup& NN_Policy::GetConfig() const {
		return policy_config;
	}

	void NN_Policy::SetAction(std::span<Trajectory> trajectories) const {
#if DP_TORCH_AVAILABLE
		int64_t input_dim = mdp->NumFlatFeatures();
		int64_t output_dim = mdp->NumValidActions();
		// Convert trajectories into a tensor for the neural network.
		torch::Tensor batched_inputs = torch::empty({ static_cast<int64_t>(trajectories.size()), input_dim }, torch::kFloat32);
		float* input_data_ptr = batched_inputs.data_ptr<float>();
		//note that this allready checks that all trajectories are await_action - separate check would
		//be superfluous. 
		mdp->GetFlatFeatures(trajectories, std::span<float>(input_data_ptr, trajectories.size() * input_dim));


		torch::NoGradGuard no_grad;
		torch::Tensor output_scores;
		switch (fw_type)
		{
		case DynaPlex::NN_Policy::NetworkForwardType::Tensor:
		{
			output_scores = neural_network->forward(batched_inputs);
		}
		break;
		case DynaPlex::NN_Policy::NetworkForwardType::TensorDict:
		case DynaPlex::NN_Policy::NetworkForwardType::TensorDictMask:
		{
			torch::Dict<std::string, torch::Tensor> dict;
			dict.reserve(1);
			dict.insert("obs",std::move( batched_inputs));
			if (fw_type == DynaPlex::NN_Policy::NetworkForwardType::TensorDictMask)
			{
				torch::Tensor batched_mask = torch::zeros({ static_cast<int64_t>(trajectories.size()), output_dim }, torch::kBool);
				bool* mask_data_ptr = batched_mask.data_ptr<bool>();
				mdp->GetMask(trajectories, std::span<bool>(mask_data_ptr, trajectories.size() * output_dim));
				dict.insert("mask", std::move(batched_mask));
			}
			output_scores = neural_network->forward(dict);
		}
		break;
		default:
			throw DynaPlex::Error("NN_Policy.forward : not supported");
			break;
		}

		// Pass the tensor through the neural network to get scores for actions.


		// Use MDP's SetArgMaxAction to determine the action based on the neural network's scores.
		mdp->SetArgMaxAction(trajectories, std::span<float>(output_scores.data_ptr<float>(), trajectories.size() * output_dim));
#else
		throw DynaPlex::Error("NN_Policy: Torch not available - Cannot SetAction. To make torch available, set dynaplex_enable_pytorch to true and dynaplex_pytorch_path to an appropriate path, e.g. in CMakeUserPresets.txt. ");
#endif
	}

	std::vector<int64_t> NN_Policy::GetPromisingActions(const DynaPlex::dp_State& dp_state, int64_t num_actions) const {
#if DP_TORCH_AVAILABLE
		int64_t input_dim = mdp->NumFlatFeatures();
		int64_t output_dim = mdp->NumValidActions();
		// Convert features into a tensor for the neural network.
		torch::Tensor batched_inputs = torch::empty({ static_cast<int64_t>(1), input_dim }, torch::kFloat32);
		float* input_data_ptr = batched_inputs.data_ptr<float>();

		mdp->GetFlatFeatures(dp_state, std::span<float>(input_data_ptr, input_dim));

		torch::NoGradGuard no_grad;
		torch::Tensor output_scores;
		switch (fw_type)
		{
		case DynaPlex::NN_Policy::NetworkForwardType::Tensor:
		{
			output_scores = neural_network->forward(batched_inputs);
		}
		break;
		case DynaPlex::NN_Policy::NetworkForwardType::TensorDict:
		case DynaPlex::NN_Policy::NetworkForwardType::TensorDictMask:
		{
			torch::Dict<std::string, torch::Tensor> dict;
			dict.reserve(1);
			dict.insert("obs", std::move(batched_inputs));
			if (fw_type == DynaPlex::NN_Policy::NetworkForwardType::TensorDictMask)
			{
				torch::Tensor batched_mask = torch::zeros({ static_cast<int64_t>(1), output_dim }, torch::kBool);
				bool* mask_data_ptr = batched_mask.data_ptr<bool>();
				mdp->GetMask(dp_state, std::span<bool>(mask_data_ptr, output_dim));
				dict.insert("mask", std::move(batched_mask));
			}
			output_scores = neural_network->forward(dict);
		}
		break;
		default:
			throw DynaPlex::Error("NN_Policy.forward : not supported");
			break;
		}

		// Pass the tensor through the neural network to get scores for actions.


		// Use MDP's GetPromisingActions to determine the promising actions based on the neural network's scores.
		std::vector<int64_t> vec = mdp->GetPromisingActions(dp_state, std::span<float>(output_scores.data_ptr<float>(), output_dim), num_actions);
		return vec;

#else
		throw DynaPlex::Error("NN_Policy: Torch not available - Cannot GetPromisingActions. To make torch available, set dynaplex_enable_pytorch to true and dynaplex_pytorch_path to an appropriate path, e.g. in CMakeUserPresets.txt. ");
#endif

	}

}  // namespace DynaPlex
