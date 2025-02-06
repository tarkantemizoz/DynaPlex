#include <iostream>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;
int main() {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup config;
	config.Add("id", "driver_assignment");
	config.Add("number_drivers", 5);
	config.Add("mean_order_per_time_unit", 0.05);
	config.Add("max_x", 55.0);
	config.Add("max_y", 6.0);
	config.Add("max_working_hours", 780.0);
	config.Add("horizon_length", 1440.0);
	config.Add("cost_per_km", 15.0);
	config.Add("penalty_for_workhour_violation", 50.0);


	DynaPlex::MDP mdp = dp.GetMDP(config);

	auto policy = mdp->GetPolicy("greedy_policy");

	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",15},
		{"mini_batch_size", 256},
		{"max_training_epochs", 100},
		{"train_based_on_probs", false}
	};

	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},//mlp - multi-layer-perceptron. 
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
	};
	int64_t num_gens = 1;

	DynaPlex::VarGroup dcl_config{
		//use defaults everywhere. 
		{"N",500000},//number of samples
		{"num_gens",num_gens},//number of neural network generations. default 5000
		{"M",500},//rollouts per action, default is 1000. 
		{"H",100},//horizon, i.e. number of steps for each rollout.
		{"L",0 },
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"enable_sequential_halving",true},
		//{"resume_gen",0}
	};


	auto dcl = dp.GetDCL(mdp, policy, dcl_config);
	dcl.TrainPolicy();
	auto policies = dcl.GetPolicies();

	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 0);
	test_config.Add("rng_seed", 10061994);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);

	auto comparer = dp.GetPolicyComparer(mdp, test_config);
	auto comparison = comparer.Compare(policies);
	for (auto& VarGroup : comparison)
	{
		std::cout << VarGroup.Dump() << std::endl;
	}

}