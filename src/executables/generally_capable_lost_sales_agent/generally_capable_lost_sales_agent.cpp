﻿#include <iostream>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include <cmath>

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

int64_t FindCOLevel(DynaPlex::VarGroup& config)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 100);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);

	auto comparer = dp.GetPolicyComparer(mdp, test_config);
	double bestCOcost = std::numeric_limits<double>::infinity();
	int64_t COLevel = 0;
	int64_t bestCOLevel = COLevel;
	double max_period_demand = 0.0;
	std::vector<double> mean_demand;
	config.Get("mean_demand", mean_demand);
	for (int64_t i = 0; i < mean_demand.size(); i++) {
		if (max_period_demand > mean_demand[i])
			max_period_demand = mean_demand[i];
	}

	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "constant_order");
	policy_config.Add("co_level", COLevel);

	while ((double) COLevel < max_period_demand)
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
	test_config.Add("warmup_periods", 100);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
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

		if (innerCBScost < bestCBScost){
			bestCBScost = innerCBScost;
			bestBSLevel = bs;
			bestCapLevel = innerBestCap;
		}
	}

	return { bestBSLevel, bestCapLevel };
}

std::pair<int64_t, int64_t> ReturnBounds(double fractile, std::vector<double> LeadTimeProbs, std::vector<int64_t> DemandCycles, std::vector<double> MeanDemands, std::vector<double> StdDemands) {
	bool found_min = false;
	double total_prob = 0.0;
	int64_t min_leadtime = LeadTimeProbs.size() - 1;
	int64_t max_leadtime = LeadTimeProbs.size() - 1;
	for (int64_t i = 0; i < LeadTimeProbs.size(); i++) {
		double prob = LeadTimeProbs[i];
		total_prob += prob;
		if (!found_min && prob > 0.0) {
			min_leadtime = i;
			found_min = true;
		}
		if (std::abs(total_prob - 1.0) < 1e-8) {
			max_leadtime = i;
			break;
		}
	}

	std::vector<double> probs_vec(LeadTimeProbs.begin() + min_leadtime, LeadTimeProbs.begin() + max_leadtime + 1);

	int64_t MaxOrderSize = 0;
	int64_t MaxSystemInv = 0;
	for (int64_t i = 0; i < MeanDemands.size(); i++) {
		std::vector<DiscreteDist> dist_vec_over_leadtime;
		dist_vec_over_leadtime.reserve(max_leadtime - min_leadtime + 1);
		std::vector<DiscreteDist> dist_vec;
		dist_vec.reserve(max_leadtime - min_leadtime + 1);
		for (int64_t j = min_leadtime; j <= max_leadtime; j++)
		{
			auto DemOverLeadtime = DiscreteDist::GetZeroDist();
			for (int64_t k = 0; k < j; k++) {
				int64_t cyclePeriod = (i + k) % DemandCycles.size();
				DynaPlex::DiscreteDist dist_over_lt = DiscreteDist::GetAdanEenigeResingDist(MeanDemands[cyclePeriod], StdDemands[cyclePeriod]);
				DemOverLeadtime = DemOverLeadtime.Add(dist_over_lt);
			}
			int64_t cyclePeriod_on_leadtime = (i + j) % DemandCycles.size();
			DynaPlex::DiscreteDist dist_on_leadtime = DiscreteDist::GetAdanEenigeResingDist(MeanDemands[cyclePeriod_on_leadtime], StdDemands[cyclePeriod_on_leadtime]);
			DemOverLeadtime = DemOverLeadtime.Add(dist_on_leadtime);
			dist_vec.push_back(dist_on_leadtime);
			dist_vec_over_leadtime.push_back(DemOverLeadtime);
		}
		auto DummyDemOverLeadtime = DiscreteDist::MultipleMix(dist_vec_over_leadtime, probs_vec);
		auto DummyDemOnLeadtime = DiscreteDist::MultipleMix(dist_vec, probs_vec);
		MaxOrderSize = std::max(MaxOrderSize, DummyDemOnLeadtime.Fractile(fractile));
		MaxSystemInv = std::max(MaxSystemInv, DummyDemOverLeadtime.Fractile(fractile));
	}

	return { MaxOrderSize,  MaxSystemInv };
}

std::vector<std::vector<double>> TestPolicies(std::vector<DynaPlex::Policy> policies, DynaPlex::VarGroup& mdp_config, DynaPlex::VarGroup& instance_config, std::vector<int64_t> periods, bool censoredProblem) {
	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 1);
	if (censoredProblem) {
		test_config.Add("warmup_periods", 0);
	}
	else {
		test_config.Add("warmup_periods", 100);
		periods = { 5000 };
	}

	DynaPlex::MDP test_mdp = dp.GetMDP(mdp_config);

	std::vector<std::vector<double>> AllPeriodResults;
	for (int64_t i = 0; i < periods.size(); i++) {
		int64_t period = periods[i];
		test_config.Set("periods_per_trajectory", period);

		dp.System() << std::endl;
		dp.System() << "Num periods:  " << period << std::endl;
		dp.System() << std::endl;

		auto comparer = dp.GetPolicyComparer(test_mdp, test_config);
		auto comparison = comparer.Compare(policies, 0, true, censoredProblem);

		double last_nn_cost = { 0.0 };
		double best_bs_cost = { 0.0 };
		double best_cbs_cost = { 0.0 };
		double initial_pol_cost = { 0.0 };
		double BSLastNNGap = { 0.0 };
		double CBSLastNNGap = { 0.0 };
		double InitialPolLastNNGap = { 0.0 };
		double BSCBSGap = { 0.0 };
		double BSInitialPolGap = { 0.0 };

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
			else if (id == "greedy_capped_base_stock") {
				VarGroup.Get("mean", initial_pol_cost);
				VarGroup.Get("mean_gap", BSInitialPolGap);
			}
			dp.System() << VarGroup.Dump() << std::endl;
		}

		if (!censoredProblem) {
			CBSLastNNGap = 100 * (last_nn_cost - best_cbs_cost) / best_cbs_cost;
			InitialPolLastNNGap = 100 * (last_nn_cost - initial_pol_cost) / initial_pol_cost;
			dp.System() << std::endl;
			dp.System() << "------------Uncensored Demand----------LowerCostBetter" << std::endl;
		}
		else {
			CBSLastNNGap = 100 * (best_cbs_cost - last_nn_cost) / best_cbs_cost;
			InitialPolLastNNGap = 100 * (initial_pol_cost - last_nn_cost) / initial_pol_cost;
			dp.System() << std::endl;
			dp.System() << "------------Censored Demand------------HigherCostBetter" << std::endl;
		}
		dp.System() << instance_config.Dump() << std::endl;
		dp.System() << "Best base-stock policy cost:  " << best_bs_cost;
		dp.System() << "  last nn_policy_cost:  " << last_nn_cost << "  gaps:  " << BSLastNNGap << "  " << CBSLastNNGap << "  " << InitialPolLastNNGap;;
		dp.System() << "  cbs cost:  " << best_cbs_cost << "  gap:  " << BSCBSGap;
		dp.System() << "  init pol cost:  " << initial_pol_cost << "  gap:  " << BSInitialPolGap << std::endl;
		dp.System() << std::endl;
		dp.System() << std::endl;

		std::vector<double> results{};
		results.push_back(best_bs_cost);
		results.push_back(best_cbs_cost);
		results.push_back(BSCBSGap);
		results.push_back(last_nn_cost);
		results.push_back(BSLastNNGap);
		results.push_back(CBSLastNNGap);
		results.push_back(initial_pol_cost);
		results.push_back(BSInitialPolGap);
		results.push_back(InitialPolLastNNGap);
		AllPeriodResults.push_back(results);
	}

	return AllPeriodResults;
}

void PrintResultsCase23(std::vector<std::vector<std::vector<std::vector<double>>>> results, size_t censoredCase, size_t period) {
	auto& dp = DynaPlexProvider::Get();

	double BSCostsAll{ 0.0 };
	double CBSCostsAll{ 0.0 };
	double BSCBSGapsAll{ 0.0 };
	double NNCostsAll{ 0.0 };
	double BSNNGapsAll{ 0.0 };
	double CBSNNGapsAll{ 0.0 };
	double InitPolCostsAll{ 0.0 };
	double BSInitPolGapsAll{ 0.0 };
	double InitPolNNGapsAll{ 0.0 };

	for (size_t k = 0; k < results.size(); k++)
	{
		BSCostsAll += results[k][censoredCase][period][0];
		CBSCostsAll += results[k][censoredCase][period][1];
		BSCBSGapsAll += results[k][censoredCase][period][2];
		NNCostsAll += results[k][censoredCase][period][3];
		BSNNGapsAll += results[k][censoredCase][period][4];
		CBSNNGapsAll += results[k][censoredCase][period][5];
		InitPolCostsAll += results[k][censoredCase][period][6];
		BSInitPolGapsAll += results[k][censoredCase][period][7];
		InitPolNNGapsAll += results[k][censoredCase][period][8];
	}
	size_t TotalNumInstanceAll = results.size();

	dp.System() << "Avg BS Costs:  " << BSCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS - CBS Gap:  " << BSCBSGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg NN Policy Costs:  " << NNCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS - NN Policy Gap:  " << BSNNGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg CBS - NN Policy Gap:  " << CBSNNGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg Init pol - NN Policy Gap:  " << InitPolNNGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS - Init Pol Gap:  " << BSInitPolGapsAll / TotalNumInstanceAll << std::endl;
}

void Case3Results(DynaPlex::VarGroup& mdp_config, std::string nn_loc, int64_t nn_num_gen) {

	auto& dp = DynaPlexProvider::Get();

	std::vector<double> p_values = { 9.0, 39.0, 69.0 };

	std::vector<std::vector<int64_t>> demand_cycles_vec = {
		{ 0 }, { 0, 1, 2 }, { 0, 1, 2, 3, 4 },  { 0, 1, 2, 3, 4, 5, 6 }
	};

	std::vector<std::vector<double>> mean_demand_vec = {
		 { 5.0 }, { 8.0, 10.0, 6.0 }, { 11.0, 3.0, 5.0, 8.0, 6.0 },  { 3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0 },
	};
	std::vector<std::string> std_vec_mix = { "pois", "geom", "negbinom", "binom", "pois", "geom", "negbinom" };

	std::vector<std::vector<double>> leadtime_distribution_vec = 
	{   { 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 },
		{ 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.1, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4 },
		{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2 },
		{ 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3 },
		{ 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.25, 0.25, 0.15, 0.1, 0.0 },
		{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5 },
		{ 0.0, 0.0, 0.0, 0.3, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0 }
	};

	DynaPlex::VarGroup instance_config;
	std::vector<int64_t> periods = { 200, 500, 1000, 2000, 5000 };

	mdp_config.Set("evaluate", true);
	for (bool ordercrossover : {  false, true }) {

		std::vector<std::vector<std::vector<std::vector<double>>>> Results;
		mdp_config.Set("order_crossover", ordercrossover);
		instance_config.Set("order_crossover", ordercrossover);

		for (std::vector<double> leadtime_dist : leadtime_distribution_vec) {

			mdp_config.Set("leadtime_distribution", leadtime_dist);
			instance_config.Set("leadtime_distribution", leadtime_dist);

			for (int64_t i = 0; i < demand_cycles_vec.size(); i++) {

				std::vector<int64_t> demand_cycles = demand_cycles_vec[i];
				mdp_config.Set("demand_cycles", demand_cycles);
				instance_config.Set("demand_cycles", demand_cycles);

				std::vector<double> mean_demand = mean_demand_vec[i];
				mdp_config.Set("mean_demand", mean_demand);
				instance_config.Set("mean_demand", mean_demand);

				double p_dummy = 0.3;
				std::vector<double> std_demand;
				std_demand.reserve(mean_demand.size());
				for (int64_t k = 0; k < mean_demand.size(); k++) {
					if (std_vec_mix[k] == "pois") {
						std_demand.push_back(std::sqrt(mean_demand[k]));
					}
					else if (std_vec_mix[k] == "binom") {
						int64_t n = static_cast<int64_t>(std::round(mean_demand[k] / p_dummy));
						double prob = mean_demand[k] / n;
						double var = n * prob * (1 - prob);
						std_demand.push_back(std::sqrt(var));
					}
					else if (std_vec_mix[k] == "geom") {
						double prob = 1.0 / (1.0 + mean_demand[k]);
						double var = (1 - prob) / (prob * prob);
						std_demand.push_back(std::sqrt(var));
					}
					else {
						int64_t r = static_cast<int64_t>(std::ceil(mean_demand[k] * p_dummy / (1 - p_dummy)));
						r = std::max(r, (int64_t)2);
						double prob = (double)r / (mean_demand[k] + r);
						double var = mean_demand[k] / prob;
						std_demand.push_back(std::sqrt(var));
					}
				}
				mdp_config.Set("stdDemand", std_demand);
				instance_config.Set("stdDemand", std_demand);

				for (double p : p_values) {

					mdp_config.Set("p", p);
					instance_config.Set("p", p);

					//mdp_config.Set("censoredDemand", false);
					//mdp_config.Set("censoredLeadtime", false);
					//int64_t BestBSLevel = FindBestBSLevel(mdp_config);
					//int64_t BestCOLevel = FindCOLevel(mdp_config);
					//std::pair<int64_t, int64_t> bounds = ReturnBounds(p / (p + 1.0), leadtime_dist, demand_cycles, mean_demand, std_demand);
					//std::pair<int64_t, int64_t> bestParams = FindCBSLevels(mdp_config, BestBSLevel, bounds.second, BestCOLevel, bounds.first);
					//int64_t BestSLevel = bestParams.first;
					//int64_t BestrLevel = bestParams.second;

					std::vector<bool> censoredLeadTime_vec = { false, true };
					std::vector<bool> censoredDemand_vec = { false, true };

					std::vector<std::vector<std::vector<double>>> InstanceResults;
					for (bool censoredLeadtime : censoredLeadTime_vec) {
						mdp_config.Set("censoredLeadtime", censoredLeadtime);
						instance_config.Set("censoredLeadtime", censoredLeadtime);
						for (bool censoredDemand : censoredDemand_vec) {
							mdp_config.Set("censoredDemand", censoredDemand);
							instance_config.Set("censoredDemand", censoredDemand);
							bool censored = (censoredDemand || censoredLeadtime) ? true : false;
	
							DynaPlex::MDP test_mdp = dp.GetMDP(mdp_config);
							DynaPlex::VarGroup policy_config;
							std::vector<DynaPlex::Policy> policies;

							//policy_config.Add("id", "base_stock");
							//policy_config.Add("base_stock_level", BestBSLevel);
							//auto best_bs_policy = test_mdp->GetPolicy(policy_config);
							//policies.push_back(best_bs_policy);

							//policy_config.Set("id", "capped_base_stock");
							//policy_config.Set("S", BestSLevel);
							//policy_config.Set("r", BestrLevel);
							//auto best_cbs_policy = test_mdp->GetPolicy(policy_config);
							//policies.push_back(best_cbs_policy);

							//auto initial_policy = test_mdp->GetPolicy("greedy_capped_base_stock");
							//policies.push_back(initial_policy);

							auto path = dp.System().filepath(nn_loc, "dcl_gen" + nn_num_gen);
							auto nn_policy = dp.LoadPolicy(test_mdp, path);
							policies.push_back(nn_policy);

							InstanceResults.push_back(TestPolicies(policies, mdp_config, instance_config, periods, censored));
						}
					}
					Results.push_back(InstanceResults);
				}
			}
		}

		// Uncensored results

		dp.System() << std::endl;
		dp.System() << "----------------Uncensored Results  " << std::endl;
		dp.System() << "---------Num periods:  " << periods.back() << std::endl;
		dp.System() << std::endl;

		PrintResultsCase23(Results, 0, 0);

		// Censored demand results

		for (size_t l = 0; l < periods.size(); l++) {
			dp.System() << std::endl;
			dp.System() << "----------------Censored Demand Results  " << std::endl;
			dp.System() << "---------Num periods:  " << periods[l] << std::endl;
			dp.System() << std::endl;

			PrintResultsCase23(Results, 1, l);
		}

		// Censored lead time results

		for (size_t l = 0; l < periods.size(); l++) {
			dp.System() << std::endl;
			dp.System() << "----------------Censored Lead Time Results  " << std::endl;
			dp.System() << "---------Num periods:  " << periods[l] << std::endl;
			dp.System() << std::endl;

			PrintResultsCase23(Results, 2, l);
		}

		// Censored demand and lead time results

		for (size_t l = 0; l < periods.size(); l++) {
			dp.System() << std::endl;
			dp.System() << "----------------Censored Demand and Lead Time Results " << std::endl;
			dp.System() << "---------Num periods:  " << periods[l] << std::endl;
			dp.System() << std::endl;

			PrintResultsCase23(Results, 3, l);
		}
	}
}

void Case2Results(DynaPlex::VarGroup& mdp_config, std::string nn_loc, int64_t nn_num_gen) {

	auto& dp = DynaPlexProvider::Get();

	std::vector<double> p_values = { 9.0, 39.0, 69.0 };
	std::vector<int64_t> leadtime_values = { 3, 6, 9 };

	std::vector<std::vector<int64_t>> demand_cycles_vec = {
		{ 0, 1 }, { 0, 1, 2 }, { 0, 1, 2, 3 },
		{ 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5, 6 }
	};

	std::vector<std::vector<std::vector<double>>> mean_demand_vec = {
		{ { 3.0, 4.0 }, { 8.0, 10.0 }, { 5.0, 7.5 } },
		{ { 2.5, 4.5, 3.0 }, { 9.0, 11.0, 9.5 }, { 3.0, 6.0, 10.0 } },
		{ { 3.0, 4.0, 2.5, 3.5 }, { 8.0, 10.0, 10.5, 9.5 }, { 10.0, 5.0, 8.0, 3.0 } },
		{ { 4.0, 2.5, 3.0, 4.5, 3.0 }, { 7.5, 11.5, 10.0, 9.5, 8.5 }, { 11.0, 3.0, 5.0, 8.0, 6.0 } },
		{ { 2.5, 4.5, 3.0, 5.0, 4.0, 3.5 }, { 10.0, 8.5, 9.0, 11.0, 9.5, 8.0 }, { 6.0, 11.5, 9.0, 4.0, 5.0, 10.0 } },
		{ { 3.0, 4.0, 2.0, 3.5, 4.5, 2.5, 4.0 }, { 11.0, 10.0, 9.0, 10.0, 11.0, 8.5, 9.5 }, { 3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0 } }
	};

	std::vector<std::string> std_vec = { "low", "high", "mix" }; //low, high, mix standard deviation of the demand
	std::vector<std::string> std_vec_low = { "pois", "binom", "pois", "binom", "pois", "binom", "pois" };
	std::vector<std::string> std_vec_high = { "geom", "negbinom", "geom", "negbinom", "geom", "negbinom", "geom" };
	std::vector<std::string> std_vec_mix = { "pois", "geom", "negbinom", "binom", "pois", "geom", "negbinom" };

	DynaPlex::VarGroup instance_config;
	std::vector<int64_t> periods = { 200, 500, 1000, 2000, 5000 };

	mdp_config.Set("evaluate", true);
	for (int64_t i = 0; i < demand_cycles_vec.size(); i++) {

		std::vector<std::vector<std::vector<std::vector<double>>>> SameCycleResults;
		std::vector<int64_t> demand_cycles = demand_cycles_vec[i];

		mdp_config.Set("demand_cycles", demand_cycles);
		instance_config.Set("demand_cycles", demand_cycles);

		for (int64_t j = 0; j < mean_demand_vec[i].size(); j++) {
			std::vector<double> mean_demand = mean_demand_vec[i][j];

			mdp_config.Set("mean_demand", mean_demand);
			instance_config.Set("mean_demand", mean_demand);

			for (std::string std_info : std_vec) {

				double p_dummy = 0.3;
				std::vector<double> std_demand;
				std_demand.reserve(mean_demand.size());
				if (std_info == "low") {
					for (int64_t k = 0; k < mean_demand.size(); k++) {
						if (std_vec_low[k] == "pois") {
							std_demand.push_back(std::sqrt(mean_demand[k]));
						}
						else {
							int64_t n = static_cast<int64_t>(std::round(mean_demand[k] / p_dummy));
							double prob = mean_demand[k] / n;
							double var = n * prob * (1 - prob);
							std_demand.push_back(std::sqrt(var));
						}
					}
				}
				else if (std_info == "high") {
					for (int64_t k = 0; k < mean_demand.size(); k++) {
						if (std_vec_high[k] == "geom") {
							double prob = 1.0 / (1.0 + mean_demand[k]);
							double var = (1 - prob) / (prob * prob);
							std_demand.push_back(std::sqrt(var));
						}
						else {
							int64_t r = static_cast<int64_t>(std::ceil(mean_demand[k] * p_dummy / (1 - p_dummy)));
							r = std::max(r, (int64_t)2);
							double prob = (double)r / (mean_demand[k] + r);
							double var = mean_demand[k] / prob;
							std_demand.push_back(std::sqrt(var));
						}
					}
				}
				else {
					for (int64_t k = 0; k < mean_demand.size(); k++) {
						if (std_vec_mix[k] == "pois") {
							std_demand.push_back(std::sqrt(mean_demand[k]));
						}
						else if (std_vec_mix[k] == "binom") {
							int64_t n = static_cast<int64_t>(std::round(mean_demand[k] / p_dummy));
							double prob = mean_demand[k] / n;
							double var = n * prob * (1 - prob);
							std_demand.push_back(std::sqrt(var));
						}
						else if (std_vec_mix[k] == "geom") {
							double prob = 1.0 / (1.0 + mean_demand[k]);
							double var = (1 - prob) / (prob * prob);
							std_demand.push_back(std::sqrt(var));
						}
						else {
							int64_t r = static_cast<int64_t>(std::ceil(mean_demand[k] * p_dummy / (1 - p_dummy)));
							r = std::max(r, (int64_t)2);
							double prob = (double)r / (mean_demand[k] + r);
							double var = mean_demand[k] / prob;
							std_demand.push_back(std::sqrt(var));
						}
					}
				}

				mdp_config.Set("stdDemand", std_demand);
				instance_config.Set("stdDemand", std_demand);

				for (double p : p_values) {

					mdp_config.Set("p", p);
					instance_config.Set("p", p);

					for (int64_t leadtime : leadtime_values) {

						int64_t max_leadtime;
						mdp_config.Get("max_leadtime", max_leadtime);
						mdp_config.Set("leadtime", leadtime);
						instance_config.Set("leadtime", leadtime);
						std::vector<double> leadtime_probs(max_leadtime + 1, 0.0);
						leadtime_probs[leadtime] = 1.0; // deterministic leadtime

						mdp_config.Set("censoredDemand", false);
						int64_t BestBSLevel = FindBestBSLevel(mdp_config);
						int64_t BestCOLevel = FindCOLevel(mdp_config);
						std::pair<int64_t, int64_t> bounds = ReturnBounds(p / (p + 1.0), leadtime_probs, demand_cycles, mean_demand, std_demand);
						std::pair<int64_t, int64_t> bestParams = FindCBSLevels(mdp_config, BestBSLevel, bounds.second, BestCOLevel, bounds.first);
						int64_t BestSLevel = bestParams.first;
						int64_t BestrLevel = bestParams.second;

						std::vector<bool> censoredDemand_vec = { false, true };
						std::vector<std::vector<std::vector<double>>> CycleResults;
						for (bool censoredDemand : censoredDemand_vec) {
							mdp_config.Set("censoredDemand", censoredDemand);
							instance_config.Set("censoredDemand", censoredDemand);

							DynaPlex::MDP test_mdp = dp.GetMDP(mdp_config);
							DynaPlex::VarGroup policy_config;
							std::vector<DynaPlex::Policy> policies;

							policy_config.Add("id", "base_stock");
							policy_config.Add("base_stock_level", BestBSLevel);
							auto best_bs_policy = test_mdp->GetPolicy(policy_config);
							policies.push_back(best_bs_policy);

							policy_config.Set("id", "capped_base_stock");
							policy_config.Set("S", BestSLevel);
							policy_config.Set("r", BestrLevel);
							auto best_cbs_policy = test_mdp->GetPolicy(policy_config);
							policies.push_back(best_cbs_policy);

							auto initial_policy = test_mdp->GetPolicy("greedy_capped_base_stock");
							policies.push_back(initial_policy);

							auto path = dp.System().filepath(nn_loc, "dcl_gen" + nn_num_gen);
							auto nn_policy = dp.LoadPolicy(test_mdp, path);
							policies.push_back(nn_policy);

							CycleResults.push_back(TestPolicies(policies, mdp_config, instance_config, periods, censoredDemand));
						}
						SameCycleResults.push_back(CycleResults);
					}
				}
			}
		}
		
		// Uncensored results

		dp.System() << std::endl;
		dp.System() << "----------------Uncensored Results With Cycle Length:  " << demand_cycles.size() << std::endl;
		dp.System() << "---------Num periods:  " << periods.back() << std::endl;
		dp.System() << std::endl;

		PrintResultsCase23(SameCycleResults, 0, 0);

		// Censored results

		for (size_t l = 0; l < periods.size(); l++) {
			dp.System() << std::endl;
			dp.System() << "----------------Censored Results With Cycle Length:  " << demand_cycles.size() << std::endl;
			dp.System() << "---------Num periods:  " << periods[l] << std::endl;
			dp.System() << std::endl;

			PrintResultsCase23(SameCycleResults, 1, l);
		}
	}
}

//void TestAll(DynaPlex::VarGroup& mdp_config, std::string nn_loc, int64_t nn_num_gen) {
//	
//	auto& dp = DynaPlexProvider::Get();
//
//	//std::vector<std::vector<int64_t>> demand_cycles_vec = { { 0 }, { 0, 1 }, { 0, 1, 2, 3, 4, 5 }, { 0, 0, 0, 0, 0, 1, 1 } };
//	//std::vector<double> mean_demand_vec = { 3.0, 5.0, 7.0, 10.0 };
//	//std::vector<double> mean_cycle_demand_vec = { 3.0, 5.0, 7.0, 10.0 };
//	//std::vector<std::string> dist_token_vec = { "binom", "poisson", "neg_binom", "geometric" };
//
//	//std::vector<bool> stochastic_leadtime_vec = { false, true };
//	//std::vector<int64_t> leadtime_values_vec = { 2, 4, 6, 8, 10 };
//	//std::vector<std::vector<double>> leadtime_distribution_vec = 
//	//{   { 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11 },
//	//	{ 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0 },
//	//	{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.1, 0.0, 0.0 },
//	//	{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4 },
//	//	{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2 },
//	//	{ 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3 },
//	//	{ 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0 },
//	//	{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.25, 0.25, 0.15, 0.1, 0.0 },
//	//	{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5 }
//	//};
//	//std::vector<bool> order_crossover_vec = { false, true };
//
//	//bool censoredDemand;
//	//bool censoredLeadTime;
//	//bool censoredRandomYield;
//	//instance_config.Get("censoredDemand", censoredDemand);
//	//instance_config.Get("censoredLeadTime", censoredLeadTime);
//	//instance_config.Get("censoredRandomYield", censoredRandomYield);
//	//mdp_config.Set("censoredDemand", censoredDemand);
//	//mdp_config.Set("censoredLeadtime", censoredLeadTime);
//	//mdp_config.Set("censoredRandomYield", censoredRandomYield);
//
//	
//	//std::vector<double> leadtime_probs;
//	//std::vector<int64_t> demand_cycles;
//	//std::vector<double> mean_demand;
//	//std::vector<double> std_demand;
//
//	mdp_config.Set("evaluate", true);
//	double p = 69.0;
//	mdp_config.Set("p", p);
//	DynaPlex::VarGroup instance_config;
//	instance_config.Add("p", p);
//
//	//////////////////////////////cyclic demand case 2
//	//double p = 39.0;
//	//int64_t leadtime = 8;
//	std::vector<int64_t> demand_cycles = { 0, 1, 2, 3, 4, 5, 6 };
//	std::vector<double> mean_demand = { 3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0 };
//	std::vector<double> std_demand = { std::sqrt(3.0), std::sqrt(5.0), std::sqrt(5.0), std::sqrt(7.0), std::sqrt(7.0), std::sqrt(10.0), std::sqrt(10.0) };
//
//	//std::vector<int64_t> demand_cycles = { 0, 1, 2, 3, 4, 5, 6 };
//	//std::vector<double> mean_demand = { 3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0 };
//	//double prob3 = 1.0 / (1.0 + 3.0);
//	//double var3 = (1 - prob3) / (prob3 * prob3);
//	//double stdev3 = std::sqrt(var3);
//	//double prob5 = 1.0 / (1.0 + 5.0);
//	//double var5 = (1 - prob5) / (prob5 * prob5);
//	//double stdev5 = std::sqrt(var5);
//	//double prob7 = 1.0 / (1.0 + 7.0);
//	//double var7 = (1 - prob7) / (prob7 * prob7);
//	//double stdev7 = std::sqrt(var7);
//	//double prob10 = 1.0 / (1.0 + 10.0);
//	//double var10 = (1 - prob10) / (prob10 * prob10);
//	//double stdev10 = std::sqrt(var10);
//	//std::vector<double> std_demand = { stdev3, stdev5, stdev5, stdev7, stdev7, stdev10, stdev10 };
//
//	//double p = 69.0;
//	//int64_t leadtime = 10;
//	//std::vector<int64_t> demand_cycles = { 0, 1, 2, 3, 4 };
//	//std::vector<double> mean_demand = { 3.0, 10.0, 5.0, 7.0, 10.0};
//	//double prob3 = 1.0 / (1.0 + 3.0);
//	//double var3 = (1 - prob3) / (prob3 * prob3);
//	//double stdev3 = std::sqrt(var3);
//	//double prob5 = 1.0 / (1.0 + 5.0);
//	//double var5 = (1 - prob5) / (prob5 * prob5);
//	//double stdev5 = std::sqrt(var5);
//	//double prob7 = 1.0 / (1.0 + 7.0);
//	//double var7 = (1 - prob7) / (prob7 * prob7);
//	//double stdev7 = std::sqrt(var7);
//	//double prob10 = 1.0 / (1.0 + 10.0);
//	//double var10 = (1 - prob10) / (prob10 * prob10);
//	//double stdev10 = std::sqrt(var10);
//	//std::vector<double> std_demand = { stdev3, std::sqrt(10.0), stdev5, stdev7, std::sqrt(10.0) };
//
//	//std::vector<int64_t> demand_cycles = { 0, 1, 2 };
//	//std::vector<double> mean_demand = { 3.0, 10.0, 5.0};
//	//double prob10 = 1.0 / (1.0 + 10.0);
//	//double var10 = (1 - prob10) / (prob10 * prob10);
//	//double stdev10 = std::sqrt(var10);
//	//std::vector<double> std_demand = { std::sqrt(3.0), stdev10, std::sqrt(5.0) };
//
//	mdp_config.Set("mean_demand", 10.0);
//	mdp_config.Set("demand_cycles", demand_cycles);
//	mdp_config.Set("mean_cylic_demands", mean_demand);
//	mdp_config.Set("std_cylic_demands", std_demand);
//	instance_config.Add("demand_cycles", demand_cycles);
//	instance_config.Add("mean_cylic_demands", mean_demand);
//	instance_config.Add("std_cylic_demands", std_demand);
//
//	//////////////////////////////cyclic demand case 2
//	// 
//	// stoch 3
//	// 39-10-pois, 69-5-geom
//	//std::vector<int64_t> demand_cycles = { 0 };
//	//double demand = 5.0;
//	//mdp_config.Set("mean_demand", demand);
//	//double prob = 1.0 / (1.0 + demand);
//	//double var = (1 - prob) / (prob * prob);
//	//double stdev = std::sqrt(var);
//	//mdp_config.Set("stdDemand", stdev);
//	//std::vector<double> mean_demand = { demand };
//	//std::vector<double> std_demand = { stdev };
//	//instance_config.Add("mean_demand", mean_demand);
//	//instance_config.Add("std_demand", std_demand);
//
//	/////////////// deterministic lead time
//	//int64_t leadtime = 10;
//	//mdp_config.Set("leadtime", leadtime);
//	//std::vector<double> leadtime_probs = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
//	//instance_config.Add("leadtime", leadtime);
//	/////////////////// deterministic lead time
//
//	/////////// stochastic lead time case 3
//
//	//std::vector<double> probs(11, 0.0);
//	//std::vector<double> leadtime_probs = probs;
//	//leadtime_probs[leadtime] = 1.0;
//	std::vector<double> leadtime_probs = { 0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.25, 0.25, 0.15, 0.1, 0.0 }; //true1
//	//std::vector<double> leadtime_probs = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; //false2
//	//std::vector<double> leadtime_probs = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4 }; //true3 cy1 p69
//	//std::vector<double> leadtime_probs = { 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0 }; //false4 cy4 p19
//
////	std::vector<double> leadtime_probs = { 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0 };
//////	{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.1, 0.0, 0.0 },
////
//	mdp_config.Set("leadtime_distribution", leadtime_probs);
//	bool order_crossover = false;
//	mdp_config.Set("order_crossover", order_crossover);
//
//	bool DeterministicLeadtime = false;
//	bool randomYield = false;
//
//	instance_config.Add("leadtime_distribution", leadtime_probs);
//	instance_config.Add("order_crossover", order_crossover);
//	instance_config.Add("randomYield", randomYield);
//
//	mdp_config.Set("randomYield", true);
//	mdp_config.Set("yield_when_realized", true);
//	mdp_config.Set("randomYield_case", 1);
//	mdp_config.Set("min_yield", 0.85);
//	mdp_config.Add("random_yield_dist",
//		DynaPlex::VarGroup({
//		{"type", "poisson"},
//		{"mean", 10.0}
//			}));
//	// instance information is set
//
//	mdp_config.Set("censoredDemand", false);
//	std::pair<int64_t, int64_t> bounds = ReturnBounds(p / (p + 1.0), leadtime_probs, demand_cycles, mean_demand, std_demand);
//	int64_t BestBSLevel = FindBestBSLevel(mdp_config);
//	int64_t BestCOLevel = FindCOLevel(mdp_config);
//	std::pair<int64_t, int64_t> bestParams = FindCBSLevels(mdp_config, BestBSLevel, bounds.second, BestCOLevel, bounds.first);
//	int64_t BestSLevel = bestParams.first;
//	int64_t BestrLevel = bestParams.second;
//
//	std::vector<bool> censoredDemand_vec = { false, true };
//	std::vector<bool> censoredLeadTime_vec = { false, true };
//	if (DeterministicLeadtime)
//		censoredLeadTime_vec = { false };
//	std::vector<bool> censoredRandomYield_vec = { false, true };
//	if (!randomYield)
//		censoredRandomYield_vec = { false };
//
//	for (bool censoredDemand : censoredDemand_vec) {
//		for (bool censoredLeadtime : censoredLeadTime_vec) {
//			for (bool censoredRandomYield : censoredRandomYield_vec) {
//
//				mdp_config.Set("censoredDemand", censoredDemand);
//				mdp_config.Set("censoredLeadtime", censoredLeadtime);
//				mdp_config.Set("censoredRandomYield", censoredRandomYield);
//				instance_config.Set("censoredDemand", censoredDemand);
//				instance_config.Set("censoredLeadtime", censoredLeadtime);
//				instance_config.Set("censoredRandomYield", censoredRandomYield);
//				bool censoredProblem = ((censoredDemand || censoredLeadtime || censoredRandomYield) ? true : false);
//
//				DynaPlex::MDP test_mdp = dp.GetMDP(mdp_config);
//				DynaPlex::VarGroup policy_config;
//				std::vector<DynaPlex::Policy> policies;
//
//				policy_config.Add("id", "base_stock");
//				policy_config.Add("base_stock_level", BestBSLevel);
//				auto best_bs_policy = test_mdp->GetPolicy(policy_config);
//				policies.push_back(best_bs_policy);
//
//				policy_config.Set("id", "capped_base_stock");
//				policy_config.Set("S", BestSLevel);
//				policy_config.Set("r", BestrLevel);
//				auto best_cbs_policy = test_mdp->GetPolicy(policy_config);
//				policies.push_back(best_cbs_policy);
//
//				auto initial_policy = test_mdp->GetPolicy("greedy_capped_base_stock");
//				policies.push_back(initial_policy);
//
//				auto path = dp.System().filepath(nn_loc, "dcl_gen" + nn_num_gen);
//				auto nn_policy = dp.LoadPolicy(test_mdp, path);
//				policies.push_back(nn_policy);
//
//				TestPoliciesNew(policies, mdp_config, instance_config, censoredProblem);
//			}
//		}
//	}
//}

void PrintResultsCase1(std::vector<std::vector<std::vector<double>>> results, size_t period) {
	auto& dp = DynaPlexProvider::Get();

	double BSCostsAll{ 0.0 };
	double TargetPolicyCostsAll{ 0.0 };
	double BSGapsAll{ 0.0 };
	double CBSGapsAll{ 0.0 };
	double CBSCostsAll{ 0.0 };
	double CBSBSGapsAll{ 0.0 };

	for (size_t i = 0; i < results.size(); i++)
	{
		BSCostsAll += results[i][period][0];
		CBSCostsAll += results[i][period][1];
		CBSBSGapsAll += results[i][period][2];
		TargetPolicyCostsAll += results[i][period][3];
		BSGapsAll += results[i][period][4];
		CBSGapsAll += results[i][period][5];
	}
	size_t TotalNumInstanceAll = results.size();

	dp.System() << "Avg BS Costs:  " << BSCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg CBS Costs:  " << CBSCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS - CBS Gap:  " << CBSBSGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg Policy Costs:  " << TargetPolicyCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS - Policy Gap:  " << BSGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg CBS - Policy Gap:  " << CBSGapsAll / TotalNumInstanceAll << std::endl;
}

void Case1Results(DynaPlex::VarGroup& config, std::string loc, int64_t num_gen, bool censored, bool paperInstances = false, bool all = true, double penalty = 5.0, int64_t tau = 7) {

	auto& dp = DynaPlexProvider::Get();

	std::vector<double> mean_demand = { 3.0, 5.0, 7.0, 10.0 };
	std::vector<std::string> dist_token = { "binom", "poisson", "neg_binom", "geometric" };
	std::vector<double> p_values = { 4.0, 9.0, 19.0, 39.0, 69.0, 99.0 };
	std::vector<int64_t> leadtime_values = { 2, 4, 6, 8, 10 };
	std::vector<int64_t> periods = { 200, 500, 1000, 2000, 5000 };

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 1000);
	config.Set("evaluate", true);
	config.Set("demand_cycles", { 0 });

	if (censored) {
		config.Set("censoredDemand", true);
		test_config.Add("warmup_periods", 0);
		if (paperInstances) {
			mean_demand = { 10.0 };
			dist_token = { "poisson", "geometric" };
			if (all) {
				p_values = { 5.0, 10.0 };
				leadtime_values = {  1, 3, 5, 7 };
			}
			else {
				p_values = { penalty };
				leadtime_values = { tau };
			}
		}
	}
	else {
		config.Set("censoredDemand", false);
		test_config.Add("warmup_periods", 100);
		periods = { 5000 };
		if (paperInstances) {
			mean_demand = { 5.0 };
			dist_token = { "poisson", "geometric" };
			p_values = { 4.0, 9.0, 19.0, 39.0 };
		}
	}

	std::vector<std::vector<std::vector<std::vector<double>>>> meandemandResults(mean_demand.size());
	std::vector<std::vector<std::vector<std::vector<double>>>> distResults(dist_token.size());
	std::vector<std::vector<std::vector<std::vector<double>>>> pResults(p_values.size());
	std::vector<std::vector<std::vector<std::vector<double>>>> leadtimeResults(leadtime_values.size());
	std::vector<std::vector<std::vector<double>>> AllResults;

	int64_t meandemandIndex = 0;
	for (double demand : mean_demand) {
		config.Set("mean_demand", { demand } );

		int64_t distIndex = 0;
		for (std::string dist : dist_token) {
			double stdev = demand;
			double p_dummy = 0.3;
			//binomial distribution
			if (dist == "binom") {
				int64_t n = static_cast<int64_t>(std::round(demand / p_dummy));
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
				int64_t r = static_cast<int64_t>(std::ceil(demand * p_dummy / (1 - p_dummy)));
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
			config.Set("stdDemand", { stdev });
			DynaPlex::DiscreteDist demand_dist = DiscreteDist::GetAdanEenigeResingDist(demand, stdev);

			int64_t pIndex = 0;
			for (double p : p_values) {
				config.Set("p", p);

				int64_t LeadTimeIndex = 0;
				for (int64_t leadtime : leadtime_values) {
					config.Set("leadtime", leadtime);

					auto DemOverLeadtime = DiscreteDist::GetZeroDist();
					for (size_t i = 0; i <= leadtime; i++)
					{
						DemOverLeadtime = DemOverLeadtime.Add(demand_dist);
					}
					int64_t MaxOrderSize = demand_dist.Fractile(p / (p + 1.0));
					int64_t MaxSystemInv = DemOverLeadtime.Fractile(p / (p + 1.0));

					if (censored) 
						config.Set("censoredDemand", false);
					
					int64_t BestBSLevel = FindBestBSLevel(config);
					int64_t BestCOLevel = FindCOLevel(config);
					std::pair<int64_t, int64_t> bestParams = FindCBSLevels(config, BestBSLevel, MaxSystemInv, BestCOLevel, MaxOrderSize);
					int64_t BestSLevel = bestParams.first;
					int64_t BestrLevel = bestParams.second;

					if (censored) 
						config.Set("censoredDemand", true);

					DynaPlex::MDP test_mdp = dp.GetMDP(config);
					DynaPlex::VarGroup policy_config;

					policy_config.Add("id", "base_stock");
					policy_config.Add("base_stock_level", BestBSLevel);
					auto best_bs_policy = test_mdp->GetPolicy(policy_config);

					policy_config.Set("id", "capped_base_stock");
					policy_config.Set("S", BestSLevel);
					policy_config.Set("r", BestrLevel);
					auto best_cbs_policy = test_mdp->GetPolicy(policy_config);

					std::vector<DynaPlex::Policy> policies;
					policies.push_back(best_bs_policy);
					policies.push_back(best_cbs_policy);
					auto path = dp.System().filepath(loc, "dcl_gen" + num_gen);
					auto nn_policy = dp.LoadPolicy(test_mdp, path);
					policies.push_back(nn_policy);

					dp.System() << config.Dump() << std::endl;
					std::vector<std::vector<double>> AllPeriodResults;
					for (int64_t i = 0; i < periods.size(); i++) {
						int64_t period = periods[i];
						test_config.Set("periods_per_trajectory", period);

						dp.System() << std::endl;
						dp.System() << "Num periods:  " << period << std::endl;
						dp.System() << std::endl;

						auto comparer = dp.GetPolicyComparer(test_mdp, test_config);
						auto comparison = comparer.Compare(policies, 0, true, censored);

						double last_nn_cost = { 0.0 };
						double best_bs_cost = { 0.0 };
						double best_cbs_cost = { 0.0 };
						double BSLastNNGap = { 0.0 };
						double CBSLastNNGap = { 0.0 };
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
							dp.System() << VarGroup.Dump() << std::endl;
						}

						if (!censored){
							CBSLastNNGap = 100 * (last_nn_cost - best_cbs_cost) / best_cbs_cost;
							dp.System() << std::endl;
							dp.System() << "------------Uncensored----------LowerCostBetter" << std::endl;
						}
						else {
							CBSLastNNGap = 100 * (best_cbs_cost - last_nn_cost) / best_cbs_cost;
							dp.System() << std::endl;
							dp.System() << "------------Censored------------HigherCostBetter" << std::endl;
						}
						dp.System() << "Mean demand: " << demand << "  dist: " << dist << "  p: " << p << "  leadtime: " << leadtime << std::endl;
						dp.System() << "Best base-stock policy cost:  " << best_bs_cost;
						dp.System() << "  last nn_policy_cost:  " << last_nn_cost << "  gaps:  " << BSLastNNGap << "  " << CBSLastNNGap;
						dp.System() << "  cbs cost:  " << best_cbs_cost << "  gap:  " << BSCBSGap << std::endl;
						dp.System() << std::endl;
						dp.System() << std::endl;

						std::vector<double> results{};
						results.push_back(best_bs_cost);
						results.push_back(best_cbs_cost);
						results.push_back(BSCBSGap);
						results.push_back(last_nn_cost);
						results.push_back(BSLastNNGap);
						results.push_back(CBSLastNNGap);
						AllPeriodResults.push_back(results);
					}
					meandemandResults[meandemandIndex].push_back(AllPeriodResults);
					distResults[distIndex].push_back(AllPeriodResults);
					pResults[pIndex].push_back(AllPeriodResults);
					leadtimeResults[LeadTimeIndex].push_back(AllPeriodResults);
					AllResults.push_back(AllPeriodResults);

					LeadTimeIndex++;
				}
				pIndex++;
			}
			distIndex++;
		}
		meandemandIndex++;
	}

	for (size_t k = 0; k < periods.size(); k++) {
		dp.System() << std::endl;
		dp.System() << "---------Num periods:  " << periods[k] << std::endl;
		dp.System() << std::endl;


		dp.System() << std::endl;
		dp.System() << "NN Policy" << " results with " << periods[k] << " periods:  " << std::endl;
		dp.System() << std::endl;

		//All results
		PrintResultsCase1(AllResults, k);
		dp.System() << std::endl;
		dp.System() << std::endl;

		//mean demand results
		for (size_t d = 0; d < mean_demand.size(); d++)
		{
			dp.System() << "Mean demand:  " << mean_demand[d] << "  ";
			PrintResultsCase1(meandemandResults[d], k);
		}
		dp.System() << std::endl;
		dp.System() << std::endl;

		//distribution results
		for (size_t d = 0; d < dist_token.size(); d++)
		{
			dp.System() << dist_token[d] << " distribution:  " << "  ";
			PrintResultsCase1(distResults[d], k);
		}
		dp.System() << std::endl;
		dp.System() << std::endl;

		//p values results
		for (size_t d = 0; d < p_values.size(); d++)
		{
			dp.System() << "Penalty cost:  " << p_values[d] << "  ";
			PrintResultsCase1(pResults[d], k);
		}
		dp.System() << std::endl;
		dp.System() << std::endl;

		//leadtime results
		for (size_t d = 0; d < leadtime_values.size(); d++)
		{
			dp.System() << "Lead time:  " << leadtime_values[d] << "  ";
			PrintResultsCase1(leadtimeResults[d], k);
		}
		dp.System() << std::endl;
		dp.System() << std::endl;
	}
}

void TrainNetwork() {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",15},
		{"mini_batch_size", 2048},
		{"max_training_epochs", 100}
	};

	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
	};

	int64_t num_gens = 6;
	DynaPlex::VarGroup dcl_config{
		//use paper hyperparameters everywhere. 
		{"N",5000000},
		{"num_gens",num_gens},
		{"SimulateOnlyPromisingActions", true},
		{"Num_Promising_Actions", 16},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"retrain_lastgen_only", false}
	};

	std::string id = "GC-LSN-Big";
	std::string exp_num = "_tsl_tcd_large_new";
	std::string loc = id + exp_num;
	dp.System() << "Network id:  " << loc << std::endl;

	bool train = false;
	bool evaluate_paper_instances_case1 = false;
	bool evaluate_all_instances_case1 = false;
	bool evaluate_all_instances_case2 = false;
	bool evaluate_all_instances_case3 = true;
	
	DynaPlex::VarGroup config;
	config.Add("id", "Zero_Shot_Lost_Sales_Inventory_Control");
	config.Add("evaluate", false);
	config.Add("train_stochastic_leadtimes", true);
	config.Add("train_cyclic_demand", true);
	config.Add("train_random_yield", false);
	config.Add("discount_factor", 1.0);
	config.Add("max_demand", 12.0);
	config.Add("max_p", 100.0);
	config.Add("max_leadtime", 10);
	config.Add("max_num_cycles", 7);

	if (train) {
		DynaPlex::MDP mdp = dp.GetMDP(config);
		auto policy = mdp->GetPolicy("greedy_capped_base_stock");
		auto dcl = dp.GetDCL(mdp, policy, dcl_config);
		dcl.TrainPolicy();

		for (int64_t gen = 1; gen <= num_gens; gen++)
		{
			auto policy = dcl.GetPolicy(gen);
			auto path = dp.System().filepath(loc, "dcl_gen" + gen);
			dp.SavePolicy(policy, path);
		}
	}

	if (dp.System().WorldRank() == 0 && evaluate_paper_instances_case1)
	{
		Case1Results(config, loc, num_gens, true, true);
		Case1Results(config, loc, num_gens, false, true);
	}

	if (dp.System().WorldRank() == 0 && evaluate_all_instances_case1)
	{
		Case1Results(config, loc, num_gens, false);
		Case1Results(config, loc, num_gens, true);
	}

	if (dp.System().WorldRank() == 0 && evaluate_all_instances_case2)
	{
		Case2Results(config, loc, 5);
	}

	if (dp.System().WorldRank() == 0 && evaluate_all_instances_case3)
	{
		Case3Results(config, loc, 5);
	}
}

int main() {

	TrainNetwork();

	return 0;
}
