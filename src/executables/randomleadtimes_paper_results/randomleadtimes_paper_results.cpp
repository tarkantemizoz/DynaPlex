#include <iostream>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

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

	while (true)
	{
		auto policy = mdp->GetPolicy(policy_config);
		auto comparison = comparer.Assess(policy);
		double cost;
		comparison.Get("mean", cost);
		//std::cout << BSLevel << "  " << cost << std::endl;
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

void TestPaperInstances() {

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

	DynaPlex::VarGroup dcl_config{
		//use paper hyperparameters everywhere. 
		{"N",5000},
		{"num_gens",3},
		{"M",1000},
		{"H", 40},
		{"L", 100},
		{"nn_architecture",nn_architecture},
		{"enable_sequential_halving", true},
		{"nn_training",nn_training}
	};

	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 100);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 18071994);

	DynaPlex::VarGroup mdp_config;
	mdp_config.Add("id", "random_leadtimes");
	mdp_config.Add("InitialInventoryLevel", 0);
	mdp_config.Add("InitialOrdersInPipeline", 0);

	bool train = true;
	bool evaluate = true;

	// Exponential distribution results
	mdp_config.Set("lead_time_dist", "exponential");
	mdp_config.Set("ExponentialLeadTimeRate", 0.5);
	mdp_config.Set("h", 1.0);
	mdp_config.Set("b", 1.0);

	for (double DemandRate : { 10.0, 5.0, 1.0 })
	{
		mdp_config.Set("DemandRate", DemandRate);
		mdp_config.Set("optimized", false);
		int64_t BestBSLevel = FindBestBSLevel(mdp_config);
		mdp_config.Set("optimized", true);
		mdp_config.Set("BaseStockLevel", BestBSLevel);
		//mdp_config.Set("InitialInventoryLevel", BestBSLevel);
		DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

		DynaPlex::VarGroup policy_config;
		policy_config.Add("id", "base_stock");
		policy_config.Add("base_stock_level", BestBSLevel);
		auto best_bs_policy = mdp->GetPolicy(policy_config);

		std::vector<DynaPlex::Policy> policies;
		policies.push_back(best_bs_policy);
		auto init_pol = mdp->GetPolicy("initial_policy");
		auto dcl = dp.GetDCL(mdp, init_pol, dcl_config);
		if (train)
		{
			dp.System() << mdp_config.Dump() << std::endl;
			dcl.TrainPolicy();
			dp.System() << std::endl;
		}

		if (evaluate)
		{
			dp.System() << mdp_config.Dump() << std::endl;
			auto dcl_policies = dcl.GetPolicies();
			for (auto policy : dcl_policies) {
				policies.push_back(policy);
			}
			auto comparer = dp.GetPolicyComparer(mdp, test_config);
			auto comparison = comparer.Compare(policies, 0);
			for (auto& VarGroup : comparison)
			{
				dp.System() << VarGroup.Dump() << std::endl;
			}
			dp.System() << std::endl;
		}
	}

	mdp_config.Set("DemandRate", 10.0);
	for (std::string case_id : { "holding", "backorder", "extreme" })
	{
		for (double c : { 3.0, 6.0, 9.0, 19.0, 39.0, 69.0, 99.0 })
		{
			bool continue_experiment = false;
			double h{ 1.0 };
			double b{ 1.0 };

			if (case_id == "holding") {
				if (c < 19.0) {
					continue_experiment = true;
					h = c;
				}
			}
			else if (case_id == "backorder") {
				if (c < 19.0) {
					continue_experiment = true;
					b = c;
				}
			}
			else {
				continue_experiment = true;
				b = c;
			}

			if (continue_experiment) {
				mdp_config.Set("h", h);
				mdp_config.Set("b", b);
				mdp_config.Set("optimized", false);
				int64_t BestBSLevel = FindBestBSLevel(mdp_config);
				mdp_config.Set("optimized", true);
				mdp_config.Set("BaseStockLevel", BestBSLevel);
				//mdp_config.Set("InitialInventoryLevel", BestBSLevel);
				DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

				DynaPlex::VarGroup policy_config;
				policy_config.Add("id", "base_stock");
				policy_config.Add("base_stock_level", BestBSLevel);
				auto best_bs_policy = mdp->GetPolicy(policy_config);
				std::vector<DynaPlex::Policy> policies;
				policies.push_back(best_bs_policy);
				auto init_pol = mdp->GetPolicy("initial_policy");
				auto dcl = dp.GetDCL(mdp, init_pol, dcl_config);

				if (train)
				{
					dp.System() << mdp_config.Dump() << std::endl;
					dcl.TrainPolicy();
					dp.System() << std::endl;
				}

				if (evaluate)
				{
					dp.System() << mdp_config.Dump() << std::endl;
					auto dcl_policies = dcl.GetPolicies();
					for (auto policy : dcl_policies) {
						policies.push_back(policy);
					}
					auto comparer = dp.GetPolicyComparer(mdp, test_config);
					auto comparison = comparer.Compare(policies, 0);
					for (auto& VarGroup : comparison)
					{
						dp.System() << VarGroup.Dump() << std::endl;
					}
					dp.System() << std::endl;
				}
			}
		}
	}

	// Uniform - pareto distribution results

	mdp_config.Set("h", 1.0);
	for (std::string dist : { "uniform", "pareto" })
	{
		if (dist == "uniform") {
			mdp_config.Set("lead_time_dist", "uniform");
			mdp_config.Set("UniformStart", 0.0);
			mdp_config.Set("UniformEnd", 4.0);
		}
		else {
			mdp_config.Set("lead_time_dist", "pareto");
			mdp_config.Set("q", 3.0);
			mdp_config.Set("tau", 0.25);
		}

		for (double DemandRate : { 10.0, 5.0, 1.0 })
		{
			for (double b : { 1.0, 9.0, 19.0, 39.0, 69.0, 99.0 })
			{
				if (!(DemandRate != 10.0 && b != 1.0))
				{
					mdp_config.Set("DemandRate", DemandRate);
					mdp_config.Set("b", b);
					mdp_config.Set("optimized", false);
					int64_t BestBSLevel = FindBestBSLevel(mdp_config);
					mdp_config.Set("optimized", true);
					mdp_config.Set("BaseStockLevel", BestBSLevel);
					//mdp_config.Set("InitialInventoryLevel", BestBSLevel);
					DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

					DynaPlex::VarGroup policy_config;
					policy_config.Add("id", "base_stock");
					policy_config.Add("base_stock_level", BestBSLevel);
					auto best_bs_policy = mdp->GetPolicy(policy_config);
					std::vector<DynaPlex::Policy> policies;
					policies.push_back(best_bs_policy);
					auto init_pol = mdp->GetPolicy("initial_policy");
					auto dcl = dp.GetDCL(mdp, init_pol, dcl_config);

					if (train)
					{
						dp.System() << mdp_config.Dump() << std::endl;
						dcl.TrainPolicy();
						dp.System() << std::endl;
					}

					if (evaluate)
					{
						dp.System() << mdp_config.Dump() << std::endl;
						auto dcl_policies = dcl.GetPolicies();
						for (auto policy : dcl_policies) {
							policies.push_back(policy);
						}
						auto comparer = dp.GetPolicyComparer(mdp, test_config);
						auto comparison = comparer.Compare(policies, 0);
						for (auto& VarGroup : comparison)
						{
							dp.System() << VarGroup.Dump() << std::endl;
						}
						dp.System() << std::endl;
					}
				}
			}
		}
	}
}

int main() {

	TestPaperInstances();

	return 0;
}