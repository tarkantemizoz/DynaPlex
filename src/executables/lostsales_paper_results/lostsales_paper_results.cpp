#include <iostream>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

//Find the best base-stock level
int64_t FindBestBSLevel(DynaPlex::VarGroup& config)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 100);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);

	auto comparer = dp.GetPolicyComparer(mdp, test_config);
	double bestBScost = std::numeric_limits<double>::infinity();
	int64_t BSLevel = 1;
	int64_t bestBSLevel = BSLevel;

	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "base_stock");
	policy_config.Add("base_stock_level", BSLevel);
	policy_config.Add("capped", false);

	while (true)
	{
		auto policy = mdp->GetPolicy(policy_config);
		auto comparison = comparer.Assess(policy);
		double cost;
		comparison.Get("mean", cost);
		if (cost < bestBScost)
		{
			bestBScost = cost;
			bestBSLevel = BSLevel;
			BSLevel++;
			policy_config.Set("base_stock_level", BSLevel);
		}
		else {
			break;
		}
	}

	return bestBSLevel;
}

void TestPaperInstances(int64_t rng_seed) {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",15},
		{"mini_batch_size", 64},
		{"max_training_epochs", 1000},
		{"train_based_on_probs", false}
	};

	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
	};

	int64_t num_gens = 3;
	int64_t N = 5000;
	bool enable_seq_halving = true;
	DynaPlex::VarGroup dcl_config{
		//use paper hyperparameters everywhere. 
		{"N",N},
		{"num_gens",num_gens},
		{"M",1000},
		{"H", 40},
		{"L", 100},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"enable_sequential_halving", enable_seq_halving},
		{"rng_seed",rng_seed}
	};

	DynaPlex::VarGroup config;
	//retrieve MDP registered under the id string "lost_sales":
	std::string id = "lost_sales";
	config.Add("id", id);
	config.Add("h", 1.0);

	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 100);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);

	DynaPlex::VarGroup exact_config = DynaPlex::VarGroup{ {"max_states",10000000}, {"silent", true } };

	std::vector<double> p_values = { 4.0, 9.0, 19.0, 39.0 };
	std::vector<int> leadtime_values = { 2, 3, 4, 6, 8, 10 };
	std::vector<std::string> demand_dist_types = { "poisson", "geometric" };

	// Training 
	for (const std::string& type : demand_dist_types) {
		for (double p : p_values) {
			for (int64_t leadtime : leadtime_values) {
				config.Set("p", p);
				config.Set("leadtime", leadtime);
				config.Set("demand_dist", DynaPlex::VarGroup({
					{"type", type},
					{"mean", 5.0}  
					}));

				DynaPlex::MDP mdp = dp.GetMDP(config);
				auto policy = mdp->GetPolicy("base_stock");

				//only print on this node in case of multi-node program. 
				dp.System() << config.Dump() << std::endl;

				// Call and train DCL with specified instance to solve
				auto dcl = dp.GetDCL(mdp, policy, dcl_config);
				dcl.TrainPolicy();

				std::string penalty_str = "_p" + std::to_string(static_cast<int64_t>(p));
				std::string leadtime_str = "_l" + std::to_string(leadtime);
				std::string numsamples_str = "_N" + std::to_string(N);
				std::string loc = id + type + penalty_str + leadtime_str + numsamples_str;
				if (!enable_seq_halving) {
					loc = loc + "_DCL0";
				}
				dp.System() << "Network id:  " << loc << std::endl;

				for (int64_t gen = 1; gen <= num_gens; gen++)
				{
					auto policy = dcl.GetPolicy(gen);
					auto path = dp.System().filepath("DeepControlledLearning", "PolicyWeights", loc, "dcl_gen" + gen);
					dp.SavePolicy(policy, path);
				}
			}
		}	
	}

	// Evaluating
	double dcl_opt_gaps = 0.0;
	double dcl_bs_gaps_all = 0.0;
	double dcl_bs_gaps_large = 0.0;

	for (const std::string& type : demand_dist_types) {
		for (double p : p_values) {
			for (int64_t leadtime : leadtime_values) {
				config.Set("p", p);
				config.Set("leadtime", leadtime);
				config.Set("demand_dist", DynaPlex::VarGroup({
					{"type", type},
					{"mean", 5.0}
					}));
				config.Set("bound_order_size_to_max_system_inv", false);

				DynaPlex::MDP mdp = dp.GetMDP(config);

				//only print on this node in case of multi-node program. 
				dp.System() << config.Dump() << std::endl;
				std::string penalty_str = "_p" + std::to_string(static_cast<int64_t>(p));
				std::string leadtime_str = "_l" + std::to_string(leadtime);
				std::string numsamples_str = "_N" + std::to_string(N);
				std::string loc = id + type + penalty_str + leadtime_str + numsamples_str;
				if (!enable_seq_halving) {
					loc = loc + "_DCL0";
				}
				dp.System() << "MDP id:  " << loc << std::endl;

				std::vector<DynaPlex::Policy> policies;
				for (int64_t gen = 1; gen <= num_gens; gen++)
				{
					auto path = dp.System().filepath("DeepControlledLearning", "PolicyWeights", loc, "dcl_gen" + gen);
					auto nn_policy = dp.LoadPolicy(mdp, path);
					policies.push_back(nn_policy);
				}

				double best_nn_cost = std::numeric_limits<double>::infinity();
				if (leadtime <= 4) {
					auto exactsolver_nn = dp.GetExactSolver(mdp, exact_config);
					for (int64_t i = 0; i < policies.size(); i++) {
						auto nn_policy = policies[i];
						double nn_pol_cost = exactsolver_nn.ComputeCosts(nn_policy);
						if (nn_pol_cost < best_nn_cost)
							best_nn_cost = nn_pol_cost;
					}

					config.Set("bound_order_size_to_max_system_inv", true);
					int64_t BestBSLevel = FindBestBSLevel(config);
					DynaPlex::VarGroup policy_config;
					policy_config.Add("id", "base_stock");
					policy_config.Add("base_stock_level", BestBSLevel);
					policy_config.Add("capped", false);
					DynaPlex::MDP mdp_test = dp.GetMDP(config);
					auto best_bs_policy = mdp_test->GetPolicy(policy_config);

					auto exactsolver = dp.GetExactSolver(mdp_test, exact_config);
					double optimal_cost = exactsolver.ComputeCosts();
					double bs_pol_cost = exactsolver.ComputeCosts(best_bs_policy);
					double bs_opt_gap = 100 * (bs_pol_cost - optimal_cost) / optimal_cost;

					double nn_opt_gap = 100 * (best_nn_cost - optimal_cost) / optimal_cost;
					dcl_opt_gaps += nn_opt_gap;
					double nn_bs_gap = 100 * (best_nn_cost - bs_pol_cost) / bs_pol_cost;
					dcl_bs_gaps_all += nn_bs_gap;
					dp.System() << "Base stock opt gap:  " << bs_opt_gap << "  DCL opt gap:  " << nn_opt_gap << "  DCL BS gap:  " << nn_bs_gap << std::endl;
					dp.System() << std::endl;
					dp.System() << std::endl;
				}
				else {
					auto comparer_nn = dp.GetPolicyComparer(mdp, test_config);
					auto comparison_nn = comparer_nn.Compare(policies);

					double best_nn_cost = std::numeric_limits<double>::infinity();
					for (auto& VarGroup : comparison_nn)
					{
						double nn_cost;
						VarGroup.Get("mean", nn_cost);
						if (nn_cost < best_nn_cost) {
							best_nn_cost = nn_cost;
						}
					}

					config.Set("bound_order_size_to_max_system_inv", true);
					int64_t BestBSLevel = FindBestBSLevel(config);
					DynaPlex::VarGroup policy_config;
					policy_config.Add("id", "base_stock");
					policy_config.Add("base_stock_level", BestBSLevel);
					policy_config.Add("capped", false);
					DynaPlex::MDP mdp_test = dp.GetMDP(config);
					auto best_bs_policy = mdp_test->GetPolicy(policy_config);

					double best_bs_cost{ 0.0 };
					auto comparer = dp.GetPolicyComparer(mdp_test, test_config);
					policies.push_back(best_bs_policy);
					auto comparison = comparer.Assess(best_bs_policy);
					comparison.Get("mean", best_bs_cost);

					double nn_bs_gap = 100 * (best_nn_cost - best_bs_cost) / best_bs_cost;
					dcl_bs_gaps_all += nn_bs_gap;
					dcl_bs_gaps_large += nn_bs_gap;
					dp.System() << "BS cost:  " << best_bs_cost << "  DCL cost:  " << best_nn_cost << "  DCL BS gap:  " << nn_bs_gap << std::endl;
					dp.System() << std::endl;
					dp.System() << std::endl;
				}
			}
		}
	}

	dcl_opt_gaps /= 24.0;
	dcl_bs_gaps_all /= 48.0;
	dcl_bs_gaps_large /= 24.0;

	dp.System() << "DCL avg opt gap:  " << dcl_opt_gaps << "  DCL avg bs gap all:  " << dcl_bs_gaps_all << "  DCL avg bs gap large:  " << dcl_bs_gaps_large << std::endl;
}

int main() {

	TestPaperInstances(10061994);

	return 0;
}