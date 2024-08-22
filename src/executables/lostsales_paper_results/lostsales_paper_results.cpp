﻿#include <iostream>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

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
	DynaPlex::VarGroup dcl_config{
		//use paper hyperparameters everywhere. 
		{"N",5000},
		{"num_gens",num_gens},
		{"M",1000},
		{"H", 40},
		{"L", 100},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"enable_sequential_halving", true},
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

	std::vector<double> p_values = { 4.0, 9.0, 19.0, 39.0 };
	std::vector<int> leadtime_values = { 2, 3, 4, 6, 8, 10 };
	std::vector<std::string> demand_dist_types = { "poisson", "geometric" };

	size_t num_exp = p_values.size() * leadtime_values.size() * demand_dist_types.size();
	std::vector<DynaPlex::VarGroup> varGroupsMDPs;
	varGroupsMDPs.reserve(num_exp);
	std::vector<std::vector<DynaPlex::VarGroup>> varGroupsPolicies_Mean;
	varGroupsPolicies_Mean.reserve(num_exp);
	std::vector<std::vector<DynaPlex::VarGroup>> varGroupsPolicies_Benchmark;
	varGroupsPolicies_Benchmark.reserve(num_exp);

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

				std::string penalty_str = "_p" + std::to_string(p);
				std::string leadtime_str = "_l" + std::to_string(leadtime);
				std::string loc = id + type + penalty_str + leadtime_str;
				dp.System() << "Network id:  " << loc << std::endl;

				for (int64_t gen = 1; gen <= num_gens; gen++)
				{
					auto policy = dcl.GetPolicy(gen);
					auto path = dp.System().filepath(loc, "dcl_gen" + gen);
					dp.SavePolicy(policy, path);
				}

				auto policies = dcl.GetPolicies();
				auto comparer = dp.GetPolicyComparer(mdp, test_config);

				varGroupsMDPs.push_back(config);
				varGroupsPolicies_Mean.push_back(comparer.Compare(policies, 0, true));
				varGroupsPolicies_Benchmark.push_back(comparer.Compare(policies));
			}
		}	
	}

	for (auto& VarGroup : varGroupsMDPs)
	{
		dp.System() << std::endl;
		dp.System() << VarGroup.Dump() << std::endl;
		for (auto& VarGroupPolicy : varGroupsPolicies_Mean.front())
		{
			dp.System() << VarGroupPolicy.Dump() << std::endl;
		}
		varGroupsPolicies_Mean.erase(varGroupsPolicies_Mean.begin());
		for (auto& VarGroupPolicy : varGroupsPolicies_Benchmark.front())
		{
			dp.System() << VarGroupPolicy.Dump() << std::endl;
		}
		varGroupsPolicies_Benchmark.erase(varGroupsPolicies_Benchmark.begin());
		dp.System() << std::endl;
	}
}

int main() {

	TestPaperInstances(10061994);

	return 0;

}