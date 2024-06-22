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

void TestLargeDemandNetwork() {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup config;
	config.Add("id", "lost_sales_one_network_extended");
	config.Add("max_p", 99.0);
	config.Add("max_leadtime", 10);
	config.Add("evaluate", true);
	config.Add("discount_factor", 1.0);
	config.Add("max_demand", 20.0);
	config.Add("min_demand", 2.0);

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 100);
	test_config.Add("periods_per_trajectory", 10000);

	std::vector<double> mean_demand = { 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0 };
	std::vector<std::string> dist_token = { "binom", "poisson", "neg_binom", "geometric" };
	std::vector<double> p_values = { 4.0, 9.0, 19.0, 39.0, 69.0, 99.0 };
	std::vector<int64_t> leadtime_values = { 2, 3, 4, 6, 8, 10 };
	size_t num_exp = p_values.size() * leadtime_values.size() * mean_demand.size() * dist_token.size();

	std::vector<DynaPlex::VarGroup> varGroupsMDPs;
	varGroupsMDPs.reserve(num_exp);
	std::vector<std::vector<DynaPlex::VarGroup>> varGroupsPolicies;
	varGroupsPolicies.reserve(num_exp);

	std::vector<std::vector<std::vector<double>>> meandemandResults(mean_demand.size());
	std::vector<std::vector<std::vector<double>>> distResults(dist_token.size());
	std::vector<std::vector<std::vector<double>>> pResults(p_values.size());
	std::vector<std::vector<std::vector<double>>> leadtimeResults(leadtime_values.size());
	std::vector<std::vector<double>> AllResults;

	int64_t meandemandIndex = 0;
	for (double demand : mean_demand) {
		config.Set("mean_demand", demand);

		int64_t distIndex = 0;
		for (std::string dist : dist_token) {
			double stdev;
			double p_dummy = 0.2;
			//binomial distribution
			if (dist == "binom") {
				int64_t n = static_cast<int64_t>(round(demand / p_dummy));
				double prob = demand / n;
				double var = n * prob * (1 - prob);
				stdev = std::sqrt(var);
			}
			//poisson distribution
			else if (dist == "poisson") {
				stdev = std::sqrt(demand);
			}
			//negative binomial distribution
			else if (dist == "neg_binom") {
				int64_t r = static_cast<int64_t>(ceil(demand * p_dummy / (1 - p_dummy)));
				r = std::max(r, (int64_t)2);
				double prob = (double)r / (demand + r);
				double var = demand / prob;
				stdev = std::sqrt(var);
			}
			//geometric distribution
			else if (dist == "geometric") {
				double prob = 1.0 / (1.0 + demand);
				double var = (1 - prob) / (prob * prob);
				stdev = std::sqrt(var);
			}
			config.Set("stdev_demand", stdev);
			DynaPlex::DiscreteDist demand_dist = DiscreteDist::GetAdanEenigeResingDist(demand, stdev);

			int64_t pIndex = 0;
			for (double p : p_values) {
				config.Set("p", p);

				int64_t LeadTimeIndex = 0;
				for (int64_t leadtime : leadtime_values) {
					config.Set("leadtime", leadtime);
					DynaPlex::MDP test_mdp = dp.GetMDP(config);
					DynaPlex::VarGroup policy_config;

					auto DemOverLeadtime = DiscreteDist::GetZeroDist();
					for (size_t i = 0; i <= leadtime; i++)
					{
						DemOverLeadtime = DemOverLeadtime.Add(demand_dist);
					}
					int64_t MaxOrderSize = demand_dist.Fractile(p / (p + 1.0));
					int64_t MaxSystemInv = DemOverLeadtime.Fractile(p / (p + 1.0));

					int64_t BestBSLevel = FindBestBSLevel(config);
					int64_t BestCOLevel = FindCOLevel(config);
					std::pair<int64_t, int64_t> bestParams = FindCBSLevels(config, BestBSLevel, MaxSystemInv, BestCOLevel, MaxOrderSize);
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

					size_t gen = 4;
					auto path = dp.System().filepath("dcl_lost_sales_one_network_extended_large", "dcl_gen" + gen);
					auto nnpolicy = dp.LoadPolicy(test_mdp, path);

					std::vector<DynaPlex::Policy> policies;
					policies.push_back(best_bs_policy);
					policies.push_back(best_co_policy);
					policies.push_back(best_cbs_policy);
					policies.push_back(nnpolicy);

					auto comparer = dp.GetPolicyComparer(test_mdp, test_config);
					varGroupsMDPs.push_back(config);
					auto comparison = comparer.Compare(policies, 0, true);
					varGroupsPolicies.push_back(comparison);

					double last_nn_cost = { 0.0 };
					double best_bs_cost = { 0.0 };
					double best_cop_cost = { 0.0 };
					double best_cbs_cost = { 0.0 };
					double BSLastNNGap = { 0.0 };
					double BSCopGap = { 0.0 };
					double BSCBSGap = { 0.0 };

					for (auto& VarGroup : comparison)
					{
						DynaPlex::VarGroup policy_id;
						VarGroup.Get("policy", policy_id);
						std::string id;
						policy_id.Get("id", id);

						if (id == "NN_Policy") {
							VarGroup.Get("mean", last_nn_cost);
							VarGroup.Get("mean_gap", BSLastNNGap);
						}
						else if (id == "base_stock") {
							VarGroup.Get("mean", best_bs_cost);
						}
						else if (id == "capped_base_stock") {
							VarGroup.Get("mean", best_cbs_cost);
							VarGroup.Get("mean_gap", BSCBSGap);
						}
						else if (id == "constant_order") {
							VarGroup.Get("mean", best_cop_cost);
							VarGroup.Get("mean_gap", BSCopGap);
						}
					}

					std::cout << std::endl;
					std::cout << "Mean demand: " << demand << "  dist: " << dist << "  p: " << p << "  leadtime: " << leadtime << std::endl;
					std::cout << "Best base-stock policy cost:  " << best_bs_cost;
					std::cout << "  last nn_policy_cost:  " << last_nn_cost << "  gap:  " << BSLastNNGap;
					std::cout << "  cop cost:  " << best_cop_cost << "  gap:  " << BSCopGap;
					std::cout << "  cbs cost:  " << best_cbs_cost << "  gap:  " << BSCBSGap << std::endl;
					std::cout << std::endl;

					std::vector<double> results{};
					results.push_back(best_bs_cost);
					results.push_back(last_nn_cost);
					results.push_back(BSLastNNGap);
					results.push_back(best_cbs_cost);
					results.push_back(BSCBSGap);
					results.push_back(best_cop_cost);
					results.push_back(BSCopGap);

					meandemandResults[meandemandIndex].push_back(results);
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
		meandemandIndex++;
	}

	for (auto& VarGroup : varGroupsMDPs)
	{
		std::cout << std::endl;
		std::cout << VarGroup.Dump() << std::endl;
		for (auto& VarGroupPolicy : varGroupsPolicies.front())
		{
			std::cout << VarGroupPolicy.Dump() << std::endl;
		}
		varGroupsPolicies.erase(varGroupsPolicies.begin());
		std::cout << std::endl;
	}

	std::vector<std::string> policy = { "NNPolicy", "CBS", "COP" };
	for (size_t j : { 0, 1, 2}) {

		std::cout << std::endl;
		std::cout << policy[j] << "  results: " << std::endl;
		std::cout << std::endl;

		//All results
		double BSCostsAll{ 0.0 };
		double TargetPolicyCostsAll{ 0.0 };
		double BSGapsAll{ 0.0 };

		for (size_t i = 0; i < AllResults.size(); i++)
		{
			BSCostsAll += AllResults[i][0];
			TargetPolicyCostsAll += AllResults[i][1 + j * 2];
			BSGapsAll += AllResults[i][2 + j * 2];
		}
		size_t TotalNumInstanceAll = AllResults.size();
		std::cout << "Avg BS Costs:  " << BSCostsAll / TotalNumInstanceAll;
		std::cout << "  , Avg Policy Costs:  " << TargetPolicyCostsAll / TotalNumInstanceAll;
		std::cout << "  , Avg BS - Policy Gap:  " << BSGapsAll / TotalNumInstanceAll << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;

		size_t demandcount = 0;
		for (double demand : mean_demand)
		{
			double BSCosts{ 0.0 };
			double TargetPolicyCosts{ 0.0 };
			double BSGaps{ 0.0 };

			for (size_t i = 0; i < meandemandResults[demandcount].size(); i++)
			{
				BSCosts += meandemandResults[demandcount][i][0];
				TargetPolicyCosts += meandemandResults[demandcount][i][1 + j * 2];
				BSGaps += meandemandResults[demandcount][i][2 + j * 2];
			}
			size_t TotalNumInstance = meandemandResults[demandcount].size();
			
			std::cout << "Mean demand:  " << demand;
			std::cout << "  Avg BS Costs:  " << BSCosts / TotalNumInstance;
			std::cout << "  , Avg Policy Costs:  " << TargetPolicyCosts / TotalNumInstance;
			std::cout << "  , Avg BS - Policy Gap:  " << BSGaps / TotalNumInstance << std::endl;

			demandcount++;
		}
		std::cout << std::endl;
		std::cout << std::endl;

		size_t distcount = 0;
		for (std::string dist : dist_token)
		{
			double BSCosts{ 0.0 };
			double TargetPolicyCosts{ 0.0 };
			double BSGaps{ 0.0 };

			for (size_t i = 0; i < distResults[distcount].size(); i++)
			{
				BSCosts += distResults[distcount][i][0];
				TargetPolicyCosts += distResults[distcount][i][1 + j * 2];
				BSGaps += distResults[distcount][i][2 + j * 2];
			}
			size_t TotalNumInstance = distResults[distcount].size();

			std::cout << dist << " distribution:  ";
			std::cout << "  Avg BS Costs:  " << BSCosts / TotalNumInstance;
			std::cout << "  , Avg Policy Costs:  " << TargetPolicyCosts / TotalNumInstance;
			std::cout << "  , Avg BS - Policy Gap:  " << BSGaps / TotalNumInstance << std::endl;

			distcount++;
		}
		std::cout << std::endl;
		std::cout << std::endl;

		size_t pcount = 0;
		for (double p : p_values)
		{
			double BSCosts{ 0.0 };
			double TargetPolicyCosts{ 0.0 };
			double BSGaps{ 0.0 };

			for (size_t i = 0; i < pResults[pcount].size(); i++)
			{
				BSCosts += pResults[pcount][i][0];
				TargetPolicyCosts += pResults[pcount][i][1 + j * 2];
				BSGaps += pResults[pcount][i][2 + j * 2];
			}
			size_t TotalNumInstance = pResults[pcount].size();

			std::cout << "Penalty cost:  " << p;
			std::cout << "  Avg BS Costs:  " << BSCosts / TotalNumInstance;
			std::cout << "  , Avg Policy Costs:  " << TargetPolicyCosts / TotalNumInstance;
			std::cout << "  , Avg BS - Policy Gap:  " << BSGaps / TotalNumInstance << std::endl;

			pcount++;
		}
		std::cout << std::endl;
		std::cout << std::endl;

		size_t lcount = 0;
		for (int64_t l : leadtime_values)
		{
			double BSCosts{ 0.0 };
			double TargetPolicyCosts{ 0.0 };
			double BSGaps{ 0.0 };

			for (size_t i = 0; i < leadtimeResults[lcount].size(); i++)
			{
				BSCosts += leadtimeResults[lcount][i][0];
				TargetPolicyCosts += leadtimeResults[lcount][i][1 + j * 2];
				BSGaps += leadtimeResults[lcount][i][2 + j * 2];
			}
			size_t TotalNumInstance = leadtimeResults[lcount].size();

			std::cout << "Lead time:  " << l;
			std::cout << "  Avg BS Costs:  " << BSCosts / TotalNumInstance;
			std::cout << "  , Avg Policy Costs:  " << TargetPolicyCosts / TotalNumInstance;
			std::cout << "  , Avg BS - Policy Gap:  " << BSGaps / TotalNumInstance << std::endl;

			lcount++;
		}
		std::cout << std::endl;
		std::cout << std::endl;
	}
}

int main() {

	TestLargeDemandNetwork();

	return 0;
}
