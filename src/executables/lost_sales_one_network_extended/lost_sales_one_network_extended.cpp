#include <iostream>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"

using namespace DynaPlex;

int64_t FindBestBSLevel(DynaPlex::VarGroup& config)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 100);
	test_config.Add("periods_per_trajectory", 10000);
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

int64_t FindCOLevel(DynaPlex::VarGroup& config)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 100);
	test_config.Add("periods_per_trajectory", 10000);
	test_config.Add("rng_seed", 1122);
	auto comparer = dp.GetPolicyComparer(mdp, test_config);

	double bestCOcost = std::numeric_limits<double>::infinity();
	int64_t COLevel = 0;
	int64_t bestCOLevel = COLevel;
	double mean_demand;
	config.Get("mean_demand", mean_demand);

	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "constant_order");
	policy_config.Add("co_level", COLevel);

	while (COLevel < mean_demand)
	{
		auto policy = mdp->GetPolicy(policy_config);
		auto comparison = comparer.Assess(policy);
		double cost;
		comparison.Get("mean", cost);

		if (cost < bestCOcost)
		{
			bestCOcost = cost;
			bestCOLevel = COLevel;
			COLevel++;
			policy_config.Set("co_level", COLevel);
		}
		else {
			break;
		}
	}

	return bestCOLevel;
}

std::pair<int64_t, int64_t> FindCBSLevels(DynaPlex::VarGroup& config, int64_t min_bs, int64_t max_bs, int64_t min_cap, int64_t max_cap)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 100);
	test_config.Add("periods_per_trajectory", 10000);
	test_config.Add("rng_seed", 1122);
	auto comparer = dp.GetPolicyComparer(mdp, test_config);

	double bestCBScost = std::numeric_limits<double>::infinity();
	int64_t bestBSLevel = max_bs;
	int64_t bestCapLevel = max_cap;

	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "capped_base_stock");

	for (int64_t bs = min_bs; bs <= max_bs; bs++)
	{
		double innerCBScost = std::numeric_limits<double>::infinity();
		int64_t innerBestCap{};

		for (int64_t cap = min_cap; cap <= max_cap; cap++)
		{
			policy_config.Set("S", bs);
			policy_config.Set("r", cap);
			auto policy = mdp->GetPolicy(policy_config);
			auto comparison = comparer.Assess(policy);
			double cost;
			comparison.Get("mean", cost);

			if (cost < innerCBScost) {
				innerCBScost = cost;
				innerBestCap = cap;
			}
			else {
				break;
			}
		}

		if (innerCBScost < bestCBScost) {
			bestCBScost = innerCBScost;
			bestBSLevel = bs;
			bestCapLevel = innerBestCap;
		}

	}

	return { bestBSLevel, bestCapLevel };
}

void runDCL() {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",15},
		{"mini_batch_size", 256},
		{"max_training_epochs", 100},
		{"train_based_on_probs", false}
	};

	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
	};

	int64_t num_gens = 4;
	DynaPlex::VarGroup dcl_config{
		//use paper hyperparameters everywhere. 
		{"N",1000000},
		{"num_gens",num_gens},
		{"M",1000},
		{"H", 40},
		{"L", 100},
		{"reinitiate_counter", 500},
		{"SimulateOnlyPromisingActions", true},
		{"Num_Promising_Actions", 8},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"retrain_lastgen_only",false}
	};

	DynaPlex::VarGroup config;
	config.Add("id", "lost_sales_one_network_extended");
	config.Add("max_p", 99.0);
	config.Add("max_leadtime", 10);
	config.Add("evaluate", false);
	config.Add("discount_factor", 1.0);
	config.Add("max_demand", 10.0);
	config.Add("min_demand", 2.0);

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 100);
	test_config.Add("periods_per_trajectory", 10000);

	DynaPlex::MDP mdp = dp.GetMDP(config);
	auto policy = mdp->GetPolicy("greedy_capped_base_stock");

	//Create a trainer for the mdp, with appropriate configuratoin. 
	auto dcl = dp.GetDCL(mdp, policy, dcl_config);
	//this trains the policy, and saves it to disk.
	dcl.TrainPolicy();

	for (size_t gen = 1; gen <= num_gens; gen++)
	{
		auto policy = dcl.GetPolicy(gen);
		auto path = dp.System().filepath("dcl_lost_sales_one_network_extended_large_v2", "dcl_gen" + gen);
		dp.SavePolicy(policy, path);
	}

	std::vector<bool> use_poisson_dist = { true, false };
	std::vector<double> p_values = { 4.0, 9.0, 19.0, 39.0 };
	std::vector<int> leadtime_values = { 2, 3, 4, 6, 8, 10 };
	size_t num_exp = p_values.size() * leadtime_values.size() * use_poisson_dist.size();

	std::vector<DynaPlex::VarGroup> varGroupsMDPs;
	varGroupsMDPs.reserve(num_exp);
	std::vector<std::vector<DynaPlex::VarGroup>> varGroupsPolicies_Mean;
	varGroupsPolicies_Mean.reserve(num_exp);
	std::vector<std::vector<DynaPlex::VarGroup>> varGroupsPolicies_Benchmark;
	varGroupsPolicies_Benchmark.reserve(num_exp);
	config.Set("evaluate", true);

	std::vector<std::vector<std::vector<double>>> distResults(2);
	std::vector<std::vector<std::vector<double>>> pResults(4);
	std::vector<std::vector<std::vector<double>>> leadtimeResults(6);
	std::vector<std::vector<double>> AllResults;

	int64_t distIndex = 0;
	for (bool pois_dist : use_poisson_dist) {
		int64_t pIndex = 0;
		for (double p : p_values) {
			int64_t LeadTimeIndex = 0;
			for (int64_t leadtime : leadtime_values) {
				double stdev;
				config.Set("p", p);
				config.Set("leadtime", leadtime);
				if (pois_dist) {
					config.Set("mean_demand", 5.0);
					config.Set("stdev_demand", std::sqrt(5));
					stdev = std::sqrt(5);
				}
				else {
					config.Set("mean_demand", 5.0);
					config.Set("stdev_demand", std::sqrt(30));
					stdev = std::sqrt(30);
				}

				DynaPlex::DiscreteDist demand_dist = DiscreteDist::GetAdanEenigeResingDist(5.0, stdev);
				auto DemOverLeadtime = DiscreteDist::GetZeroDist();
				for (size_t i = 0; i <= leadtime; i++)
				{
					DemOverLeadtime = DemOverLeadtime.Add(demand_dist);
				}
				int64_t MaxOrderSize = demand_dist.Fractile(p / (p + 1.0));
				int64_t MaxSystemInv = DemOverLeadtime.Fractile(p / (p + 1.0));

				DynaPlex::MDP test_mdp = dp.GetMDP(config);
				DynaPlex::VarGroup policy_config;

				int64_t BestBSLevel = FindBestBSLevel(config);
				int64_t BestCOLevel = FindCOLevel(config);
				std::pair<int64_t, int64_t> bestParams;
				bestParams = FindCBSLevels(config, BestBSLevel, MaxSystemInv, BestCOLevel, MaxOrderSize);
				int64_t BestSLevel = bestParams.first;
				int64_t BestrLevel = bestParams.second;

				policy_config.Add("id", "base_stock");
				policy_config.Add("base_stock_level", BestBSLevel);
				auto best_bs_policy = test_mdp->GetPolicy(policy_config);

				policy_config.Set("id", "constant_order");
				policy_config.Set("co_level", BestCOLevel);
				auto best_co_policy = test_mdp->GetPolicy(policy_config);

				policy_config.Set("id", "capped_base_stock");
				policy_config.Set("S", BestSLevel);
				policy_config.Set("r", BestrLevel);
				auto best_cbs_policy = test_mdp->GetPolicy(policy_config);

				std::vector<DynaPlex::Policy> policies;
				policies.push_back(best_bs_policy);
				policies.push_back(best_co_policy);
				policies.push_back(best_cbs_policy);
				for (size_t gen = 1; gen <= num_gens; gen++)
				{
					auto path = dp.System().filepath("dcl_lost_sales_one_network_extended_large_v2", "dcl_gen" + gen);
					auto nn_policy = dp.LoadPolicy(test_mdp, path);
					policies.push_back(nn_policy);
				}

				auto comparer = dp.GetPolicyComparer(test_mdp, test_config);
				varGroupsMDPs.push_back(config);
				auto comparison = comparer.Compare(policies);
				varGroupsPolicies_Mean.push_back(comparer.Compare(policies));
				varGroupsPolicies_Benchmark.push_back(comparer.Compare(policies, 0));

				double best_nn_cost = std::numeric_limits<double>::infinity();
				double last_nn_cost = std::numeric_limits<double>::infinity();
				double best_bs_cost = std::numeric_limits<double>::infinity();
				double best_cop_cost = std::numeric_limits<double>::infinity();
				double best_cbs_cost = std::numeric_limits<double>::infinity();
				for (auto& VarGroup : comparison)
				{
					DynaPlex::VarGroup policy_id;
					VarGroup.Get("policy", policy_id);
					std::string id;
					policy_id.Get("id", id);

					if (id == "NN_Policy") {
						double nn_cost;
						VarGroup.Get("mean", nn_cost);
						last_nn_cost = nn_cost;
						if (nn_cost < best_nn_cost) {
							best_nn_cost = nn_cost;
						}
					}
					else if (id == "base_stock") {
						VarGroup.Get("mean", best_bs_cost);
					}
					else if (id == "capped_base_stock") {
						VarGroup.Get("mean", best_cbs_cost);
					}
					else if (id == "constant_order") {
						VarGroup.Get("mean", best_cop_cost);
					}
				}
				double BSNNGap = (100 * (best_nn_cost - best_bs_cost) / best_bs_cost);
				double BSLastNNGap = (100 * (last_nn_cost - best_bs_cost) / best_bs_cost);
				double BSCopGap = (100 * (best_cop_cost - best_bs_cost) / best_bs_cost);
				double BSCBSGap = (100 * (best_cbs_cost - best_bs_cost) / best_bs_cost);
				std::cout << std::endl;
				std::cout << "Best base-stock policy cost:  " << best_bs_cost << "  best nn-policy cost:  " << best_nn_cost << "  gap:  " << BSNNGap;
				std::cout << "  last nn_policy_cost:  " << last_nn_cost << "  gap:  " << BSLastNNGap;
				std::cout << "  cop cost:  " << best_cop_cost << "  gap:  " << BSCopGap;
				std::cout << "  cbs cost:  " << best_cbs_cost << "  gap:  " << BSCBSGap << std::endl;
				std::cout << std::endl;

				std::vector<double> results{};
				results.push_back(best_bs_cost);
				results.push_back(best_nn_cost);
				results.push_back(BSNNGap);
				results.push_back(last_nn_cost);
				results.push_back(BSLastNNGap);
				results.push_back(best_cop_cost);
				results.push_back(BSCopGap);
				results.push_back(best_cbs_cost);
				results.push_back(BSCBSGap);

				distResults[distIndex].push_back(results);
				pResults[pIndex].push_back(results);
				leadtimeResults[LeadTimeIndex].push_back(results);
				AllResults.push_back(results);

				LeadTimeIndex++;
			}
			pIndex++;
		}
		distIndex++;
	}

	
	for (auto& VarGroup : varGroupsMDPs)
	{
		std::cout << std::endl;
		std::cout << VarGroup.Dump() << std::endl;
		for (auto& VarGroupPolicy : varGroupsPolicies_Mean.front())
		{
			std::cout << VarGroupPolicy.Dump() << std::endl;
		}
		varGroupsPolicies_Mean.erase(varGroupsPolicies_Mean.begin());
		for (auto& VarGroupPolicy : varGroupsPolicies_Benchmark.front())
		{
			std::cout << VarGroupPolicy.Dump() << std::endl;
		}
		varGroupsPolicies_Benchmark.erase(varGroupsPolicies_Benchmark.begin());
		std::cout << std::endl;
	}

	for (size_t j : { 0, 2, 4, 6}) {
		//All results
		double BSCosts{ 0.0 };
		double NNCosts{ 0.0 };
		double BSNNGaps{ 0.0 };

		for (size_t i = 0; i < AllResults.size(); i++)
		{
			BSCosts += AllResults[i][0];
			NNCosts += AllResults[i][1 + j];
			BSNNGaps += AllResults[i][2 + j];
		}
		size_t TotalNumInstance = AllResults.size();
		std::cout << "Avg BS Costs:  " << BSCosts / TotalNumInstance << "  , Avg NN Costs:  " << NNCosts / TotalNumInstance;
		std::cout << "  , Avg BS - NN Gap:  " << BSNNGaps / TotalNumInstance << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;

		size_t dcount = 0;
		for (bool pois_dist : use_poisson_dist)
		{
			double pBSCosts{ 0.0 };
			double pNNCosts{ 0.0 };
			double pBSNNGaps{ 0.0 };

			for (size_t i = 0; i < distResults[dcount].size(); i++)
			{
				pBSCosts += distResults[dcount][i][0];
				pNNCosts += distResults[dcount][i][1 + j];
				pBSNNGaps += distResults[dcount][i][2 + j];
			}
			size_t pTotalNumInstance = distResults[dcount].size();
			if (pois_dist)
				std::cout << "Poisson Distribution: ";
			else
				std::cout << "Geometric Distribution:  ";
			std::cout << "Avg BS Costs:  " << pBSCosts / pTotalNumInstance;
			std::cout << "  , Avg NN Costs:  " << pNNCosts / pTotalNumInstance;
			std::cout << "  , Avg BS - NN Gap:  " << pBSNNGaps / pTotalNumInstance << std::endl;

			dcount++;
		}
		std::cout << std::endl;
		std::cout << std::endl;

		size_t pcount = 0;
		for (double p : p_values)
		{
			double pBSCosts{ 0.0 };
			double pNNCosts{ 0.0 };
			double pBSNNGaps{ 0.0 };

			for (size_t i = 0; i < pResults[pcount].size(); i++)
			{
				pBSCosts += pResults[pcount][i][0];
				pNNCosts += pResults[pcount][i][1 + j];
				pBSNNGaps += pResults[pcount][i][2 + j];
			}
			size_t pTotalNumInstance = pResults[pcount].size();
			std::cout << "Penalty cost:  " << p << "  Avg BS Costs:  " << pBSCosts / pTotalNumInstance;
			std::cout << "  , Avg NN Costs:  " << pNNCosts / pTotalNumInstance;
			std::cout << "  , Avg BS - NN Gap:  " << pBSNNGaps / pTotalNumInstance << std::endl;

			pcount++;
		}
		std::cout << std::endl;
		std::cout << std::endl;

		//Lead time results
		size_t lcount = 0;
		for (int l : leadtime_values)
		{
			double mBSCosts{ 0.0 };
			double mNNCosts{ 0.0 };
			double mBSNNGaps{ 0.0 };

			for (size_t i = 0; i < leadtimeResults[lcount].size(); i++)
			{
				mBSCosts += leadtimeResults[lcount][i][0];
				mNNCosts += leadtimeResults[lcount][i][1 + j];
				mBSNNGaps += leadtimeResults[lcount][i][2 + j];
			}
			size_t mTotalNumInstance = leadtimeResults[lcount].size();
			std::cout << "Lead time:  " << l << "  Avg BS Costs:  " << mBSCosts / mTotalNumInstance;
			std::cout << "  , Avg NN Costs:  " << mNNCosts / mTotalNumInstance;
			std::cout << "  , Avg BS - NN Gap:  " << mBSNNGaps / mTotalNumInstance << std::endl;

			lcount++;
		}
		std::cout << std::endl;
		std::cout << std::endl;
	}
}

void TestBSPolicy() {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup config;
	config.Add("id", "lost_sales_one_network_extended");
	config.Add("max_p", 99.0);
	config.Add("MaxLeadTime", 10);
	config.Add("evaluate", true);
	config.Add("discount_factor", 1.0);
	config.Set("p", 39.0);
	config.Set("leadtime", 10);
	config.Set("poisson_dist", false);

	DynaPlex::MDP test_mdp = dp.GetMDP(config);

	int64_t BestBSLevel = FindBestBSLevel(config);
	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "base_stock");
	policy_config.Add("base_stock_level", BestBSLevel);
	auto best_bs_policy = test_mdp->GetPolicy(policy_config);

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 100);
	test_config.Add("periods_per_trajectory", 10000);
	test_config.Add("rng_seed", 1122);
	auto comparer = dp.GetPolicyComparer(test_mdp, test_config);

	auto comparison = comparer.Assess(best_bs_policy);
	std::cout << comparison.Dump() << std::endl;
}

int main() {

	runDCL();

	//TestBSPolicy();

	return 0;
}
