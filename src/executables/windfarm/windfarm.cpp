#include <iostream>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;
int main() {

	auto& dp = DynaPlexProvider::Get();

	// --- MDP configuration (small instance: 2 farms x 2 windmills) -----------------------
	// Kept small (N=4 windmills) so the exact solver is tractable: it enumerates K^N joint
	// degradation events (K=6 sub-intervals here), which is 6^4=1296 per state. At 2x3 this
	// would be 6^6=46656 events x ~8192 states and is infeasible.
	// degrade_probs / jump_red_probs are indexed by current health [Blue, Yellow, Orange]:
	//   degrade_probs[i] = P(degrade one level), jump_red_probs[i] = P(jump straight to Red).
	DynaPlex::VarGroup config;
	config.Add("id", "windfarm");
	config.Add("num_farms", 2);
	config.Add("windmills_per_farm", 2);
	config.Add("degrade_probs", DynaPlex::VarGroup::DoubleVec{ 0.15, 0.20, 0.25 });
	config.Add("jump_red_probs", DynaPlex::VarGroup::DoubleVec{ 0.01, 0.03, 0.10 });
	config.Add("travel_cost", 5.0);
	config.Add("maintenance_cost", 2.0);
	config.Add("repair_cost", 10.0);
	config.Add("red_penalty", 20.0);
	config.Add("discount_factor", 1.0); // average-cost, infinite-horizon (matches the exact solver's regime)

	DynaPlex::MDP mdp = dp.GetMDP(config);

	// Greedy benchmark policy, used both as a baseline and to bootstrap DCL.
	auto policy = mdp->GetPolicy("greedy_engineer");

	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience", 15},
		{"mini_batch_size", 256},
		{"max_training_epochs", 100},
		{"train_based_on_probs", false}
	};

	DynaPlex::VarGroup nn_architecture{
		{"type", "mlp"},
		{"hidden_layers", DynaPlex::VarGroup::Int64Vec{256, 128, 128, 128}}
	};

	int64_t num_gens = 1;

	DynaPlex::VarGroup dcl_config{
		{"N", 2000},          // number of samples
		{"num_gens", num_gens}, // number of neural network generations
		{"M", 100},             // rollouts per action
		{"H", 30},             // horizon (steps per rollout)
		{"L", 100},
		{"nn_architecture", nn_architecture},
		{"nn_training", nn_training},
		{"enable_sequential_halving", true}
	};

	auto dcl = dp.GetDCL(mdp, policy, dcl_config);
	dcl.TrainPolicy();
	// GetPolicies() returns the bootstrap (greedy) policy as generation 0, followed by one
	// trained neural-network policy per DCL generation.
	auto policies = dcl.GetPolicies();

	// --- Exact optimal policy via value/policy iteration ---------------------------------
	// Feasible here because the instance is small (see note at the top). discount_factor==1.0
	// makes this an average-cost computation, which is the regime the exact solver supports.
	DynaPlex::VarGroup exact_config{ {"max_states", 1000000}, {"silent", true} };
	auto exact_solver = dp.GetExactSolver(mdp, exact_config);
	double optimal_cost = exact_solver.ComputeCosts();            // exact optimal average cost per period
	auto optimal_policy = exact_solver.GetOptimalPolicy();        // reuses the just-computed solution
	std::cout << "Exact optimal average cost per period: " << optimal_cost << std::endl;

	// Append the exact-optimal policy so it is compared on equal footing with greedy and DCL.
	policies.push_back(optimal_policy);
	const int64_t optimal_index = static_cast<int64_t>(policies.size()) - 1;

	// --- Compare greedy vs DCL generations vs exact-optimal ------------------------------
	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 100);
	test_config.Add("rng_seed", 10061994);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);

	auto comparer = dp.GetPolicyComparer(mdp, test_config);
	// Benchmark against the exact-optimal policy and report each policy's gap relative to it.
	auto comparison = comparer.Compare(policies, optimal_index, /*compute_gap=*/true);
	for (auto& VarGroup : comparison)
	{
		std::cout << VarGroup.Dump() << std::endl;
	}

	return 0;
}
