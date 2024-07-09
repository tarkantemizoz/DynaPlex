#include <iostream>
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

void PrintResults(std::vector<std::vector<std::vector<double>>> results, size_t period, size_t policy) {
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
		TargetPolicyCostsAll += results[i][period][3 + policy * 3];
		BSGapsAll += results[i][period][4 + policy * 3];
		CBSGapsAll += results[i][period][5 + policy * 3];
	}
	size_t TotalNumInstanceAll = results.size();

	dp.System() << "Avg BS Costs:  " << BSCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg CBS Costs:  " << CBSCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS - CBS Gap:  " << CBSBSGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg Policy Costs:  " << TargetPolicyCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS - Policy Gap:  " << BSGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg CBS - Policy Gap:  " << CBSGapsAll / TotalNumInstanceAll << std::endl;
}


void TestPoliciesNew(std::vector<DynaPlex::Policy> policies, DynaPlex::VarGroup& mdp_config, DynaPlex::VarGroup& instance_config, bool censoredProblem) {
	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 1000);
	std::vector<int64_t> periods = { 200, 500, 1000, 2000, 5000 };
	if (censoredProblem) {
		test_config.Add("warmup_periods", 0);
	}
	else {
		test_config.Add("warmup_periods", 100);
		periods = { 5000 };
	}

	bool returnRewards;
	mdp_config.Get("returnRewards", returnRewards);
	DynaPlex::MDP test_mdp = dp.GetMDP(mdp_config);

	std::vector<std::vector<double>> AllPeriodResults;
	for (int64_t i = 0; i < periods.size(); i++) {
		int64_t period = periods[i];
		test_config.Set("periods_per_trajectory", period);

		dp.System() << std::endl;
		dp.System() << "Num periods:  " << period << std::endl;
		dp.System() << std::endl;

		auto comparer = dp.GetPolicyComparer(test_mdp, test_config);
		auto comparison = comparer.Compare(policies, 0, true, returnRewards);

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

		if (!returnRewards) {
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
		AllPeriodResults.push_back(results);
	}
}


void TestAll(DynaPlex::VarGroup& mdp_config, std::string nn_loc, int64_t nn_num_gen) {
	
	auto& dp = DynaPlexProvider::Get();

	//std::vector<std::vector<int64_t>> demand_cycles_vec = { { 0 }, { 0, 1 }, { 0, 1, 2, 3, 4, 5 }, { 0, 0, 0, 0, 0, 1, 1 } };
	//std::vector<double> mean_demand_vec = { 3.0, 5.0, 7.0, 10.0 };
	//std::vector<double> mean_cycle_demand_vec = { 3.0, 5.0, 7.0, 10.0 };
	//std::vector<std::string> dist_token_vec = { "binom", "poisson", "neg_binom", "geometric" };

	//std::vector<bool> stochastic_leadtime_vec = { false, true };
	//std::vector<int64_t> leadtime_values_vec = { 2, 4, 6, 8, 10 };
	//std::vector<std::vector<double>> leadtime_distribution_vec = 
	//{   { 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11 },
	//	{ 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0 },
	//	{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.1, 0.0, 0.0 },
	//	{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4 },
	//	{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2 },
	//	{ 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3 },
	//	{ 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0 },
	//	{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.25, 0.25, 0.15, 0.1, 0.0 },
	//	{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5 }
	//};
	//std::vector<bool> order_crossover_vec = { false, true };

	//bool censoredDemand;
	//bool censoredLeadTime;
	//bool censoredRandomYield;
	//instance_config.Get("censoredDemand", censoredDemand);
	//instance_config.Get("censoredLeadTime", censoredLeadTime);
	//instance_config.Get("censoredRandomYield", censoredRandomYield);
	//mdp_config.Set("censoredDemand", censoredDemand);
	//mdp_config.Set("censoredLeadtime", censoredLeadTime);
	//mdp_config.Set("censoredRandomYield", censoredRandomYield);

	
	//std::vector<double> leadtime_probs;
	//std::vector<int64_t> demand_cycles;
	//std::vector<double> mean_demand;
	//std::vector<double> std_demand;

	mdp_config.Set("evaluate", true);
	double p = 69.0;
	mdp_config.Set("p", p);
	DynaPlex::VarGroup instance_config;
	instance_config.Add("p", p);

	//////////////////////////////cyclic demand case 2
	//double p = 39.0;
	//int64_t leadtime = 8;
	std::vector<int64_t> demand_cycles = { 0, 1, 2, 3, 4, 5, 6 };
	std::vector<double> mean_demand = { 3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0 };
	std::vector<double> std_demand = { std::sqrt(3.0), std::sqrt(5.0), std::sqrt(5.0), std::sqrt(7.0), std::sqrt(7.0), std::sqrt(10.0), std::sqrt(10.0) };

	//std::vector<int64_t> demand_cycles = { 0, 1, 2, 3, 4, 5, 6 };
	//std::vector<double> mean_demand = { 3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0 };
	//double prob3 = 1.0 / (1.0 + 3.0);
	//double var3 = (1 - prob3) / (prob3 * prob3);
	//double stdev3 = std::sqrt(var3);
	//double prob5 = 1.0 / (1.0 + 5.0);
	//double var5 = (1 - prob5) / (prob5 * prob5);
	//double stdev5 = std::sqrt(var5);
	//double prob7 = 1.0 / (1.0 + 7.0);
	//double var7 = (1 - prob7) / (prob7 * prob7);
	//double stdev7 = std::sqrt(var7);
	//double prob10 = 1.0 / (1.0 + 10.0);
	//double var10 = (1 - prob10) / (prob10 * prob10);
	//double stdev10 = std::sqrt(var10);
	//std::vector<double> std_demand = { stdev3, stdev5, stdev5, stdev7, stdev7, stdev10, stdev10 };

	//double p = 69.0;
	//int64_t leadtime = 10;
	//std::vector<int64_t> demand_cycles = { 0, 1, 2, 3, 4 };
	//std::vector<double> mean_demand = { 3.0, 10.0, 5.0, 7.0, 10.0};
	//double prob3 = 1.0 / (1.0 + 3.0);
	//double var3 = (1 - prob3) / (prob3 * prob3);
	//double stdev3 = std::sqrt(var3);
	//double prob5 = 1.0 / (1.0 + 5.0);
	//double var5 = (1 - prob5) / (prob5 * prob5);
	//double stdev5 = std::sqrt(var5);
	//double prob7 = 1.0 / (1.0 + 7.0);
	//double var7 = (1 - prob7) / (prob7 * prob7);
	//double stdev7 = std::sqrt(var7);
	//double prob10 = 1.0 / (1.0 + 10.0);
	//double var10 = (1 - prob10) / (prob10 * prob10);
	//double stdev10 = std::sqrt(var10);
	//std::vector<double> std_demand = { stdev3, std::sqrt(10.0), stdev5, stdev7, std::sqrt(10.0) };

	//std::vector<int64_t> demand_cycles = { 0, 1, 2 };
	//std::vector<double> mean_demand = { 3.0, 10.0, 5.0};
	//double prob10 = 1.0 / (1.0 + 10.0);
	//double var10 = (1 - prob10) / (prob10 * prob10);
	//double stdev10 = std::sqrt(var10);
	//std::vector<double> std_demand = { std::sqrt(3.0), stdev10, std::sqrt(5.0) };

	mdp_config.Set("mean_demand", 10.0);
	mdp_config.Set("demand_cycles", demand_cycles);
	mdp_config.Set("mean_cylic_demands", mean_demand);
	mdp_config.Set("std_cylic_demands", std_demand);
	instance_config.Add("demand_cycles", demand_cycles);
	instance_config.Add("mean_cylic_demands", mean_demand);
	instance_config.Add("std_cylic_demands", std_demand);

	//////////////////////////////cyclic demand case 2
	// 
	// stoch 3
	// 39-10-pois, 69-5-geom
	//std::vector<int64_t> demand_cycles = { 0 };
	//double demand = 5.0;
	//mdp_config.Set("mean_demand", demand);
	//double prob = 1.0 / (1.0 + demand);
	//double var = (1 - prob) / (prob * prob);
	//double stdev = std::sqrt(var);
	//mdp_config.Set("stdDemand", stdev);
	//std::vector<double> mean_demand = { demand };
	//std::vector<double> std_demand = { stdev };
	//instance_config.Add("mean_demand", mean_demand);
	//instance_config.Add("std_demand", std_demand);

	/////////////// deterministic lead time
	//int64_t leadtime = 10;
	//mdp_config.Set("leadtime", leadtime);
	//std::vector<double> leadtime_probs = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
	//instance_config.Add("leadtime", leadtime);
	/////////////////// deterministic lead time

	/////////// stochastic lead time case 3

	//std::vector<double> probs(11, 0.0);
	//std::vector<double> leadtime_probs = probs;
	//leadtime_probs[leadtime] = 1.0;
	//std::vector<double> leadtime_probs = { 0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.25, 0.25, 0.15, 0.1, 0.0 }; //true1
	//std::vector<double> leadtime_probs = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; //false2
	std::vector<double> leadtime_probs = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4 }; //true3 cy1 p69
	//std::vector<double> leadtime_probs = { 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0 }; //false4 cy4 p19

//	std::vector<double> leadtime_probs = { 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0 };
////	{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.1, 0.0, 0.0 },
//
	mdp_config.Set("leadtime_distribution", leadtime_probs);
	bool order_crossover = true;
	mdp_config.Set("order_crossover", order_crossover);

	bool DeterministicLeadtime = false;
	bool randomYield = true;

	instance_config.Add("leadtime_distribution", leadtime_probs);
	instance_config.Add("order_crossover", order_crossover);
	instance_config.Add("randomYield", randomYield);

	mdp_config.Set("randomYield", true);
	mdp_config.Set("yield_when_realized", true);
	mdp_config.Set("randomYield_case", 1);
	mdp_config.Set("min_yield", 0.85);
	mdp_config.Add("random_yield_dist",
		DynaPlex::VarGroup({
		{"type", "poisson"},
		{"mean", 10.0}
			}));
	// instance information is set

	mdp_config.Set("returnRewards", false);
	mdp_config.Set("collectStatistics", false);
	std::pair<int64_t, int64_t> bounds = ReturnBounds(p / (p + 1.0), leadtime_probs, demand_cycles, mean_demand, std_demand);

	int64_t BestBSLevel = FindBestBSLevel(mdp_config);
	int64_t BestCOLevel = FindCOLevel(mdp_config);
	std::pair<int64_t, int64_t> bestParams = FindCBSLevels(mdp_config, BestBSLevel, bounds.second, BestCOLevel, bounds.first);
	int64_t BestSLevel = bestParams.first;
	int64_t BestrLevel = bestParams.second;


	std::vector<bool> censoredDemand_vec = { false, true };
	std::vector<bool> censoredLeadTime_vec = { false, true };
	if (DeterministicLeadtime)
		censoredLeadTime_vec = { false };
	std::vector<bool> censoredRandomYield_vec = { false, true };
	if (!randomYield)
		censoredRandomYield_vec = { false };

	mdp_config.Set("collectStatistics", true);
	for (bool censoredDemand : censoredDemand_vec) {
		for (bool censoredLeadtime : censoredLeadTime_vec) {
			for (bool censoredRandomYield : censoredRandomYield_vec) {

				mdp_config.Set("returnRewards", (censoredDemand ? true : false));
				mdp_config.Set("censoredDemand", censoredDemand);
				mdp_config.Set("censoredLeadtime", censoredLeadtime);
				mdp_config.Set("censoredRandomYield", censoredRandomYield);
				instance_config.Set("censoredDemand", censoredDemand);
				instance_config.Set("censoredLeadtime", censoredLeadtime);
				instance_config.Set("censoredRandomYield", censoredRandomYield);
				bool censoredProblem = ((censoredDemand || censoredLeadtime || censoredRandomYield) ? true : false);

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

				TestPoliciesNew(policies, mdp_config, instance_config, censoredProblem);
			}
		}
	}
}


void TestPolicies(DynaPlex::VarGroup& config, std::string loc, int64_t num_gen, bool censored, bool paperInstances = false, bool all = true, double penalty = 5.0, int64_t tau = 7) {

	auto& dp = DynaPlexProvider::Get();

	std::vector<double> mean_demand = { 3.0, 5.0, 7.0, 10.0 };
	std::vector<std::string> dist_token = { "binom", "poisson", "neg_binom", "geometric" };
	std::vector<double> p_values = { 4.0, 9.0, 19.0, 39.0, 69.0, 99.0 };
	std::vector<int64_t> leadtime_values = { 2, 4, 6, 8, 10 };
	std::vector<int64_t> periods = { 200, 500, 1000, 2000, 5000 };

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 1000);
	config.Set("evaluate", true);
	config.Set("returnRewards", false);
	config.Set("collectStatistics", false);
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
		config.Set("mean_demand", demand);

		int64_t distIndex = 0;
		for (std::string dist : dist_token) {
			double stdev = demand;
			double p_dummy = 0.2;
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
			config.Set("stdDemand", stdev);
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

					if (censored) {
						config.Set("returnRewards", false);
						config.Set("collectStatistics", false);
					}

					int64_t BestBSLevel = FindBestBSLevel(config);
					int64_t BestCOLevel = FindCOLevel(config);
					std::pair<int64_t, int64_t> bestParams = FindCBSLevels(config, BestBSLevel, MaxSystemInv, BestCOLevel, MaxOrderSize);
					int64_t BestSLevel = bestParams.first;
					int64_t BestrLevel = bestParams.second;

					if (censored) {
						config.Set("returnRewards", true);
						config.Set("collectStatistics", true);
					}

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
					for (size_t gen = 1; gen <= num_gen; gen++)
					{
						auto path = dp.System().filepath(loc, "dcl_gen" + gen);
						auto nn_policy = dp.LoadPolicy(test_mdp, path);
						policies.push_back(nn_policy);
					}

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
						double best_nn_cost = { 0.0 };
						double best_bs_cost = { 0.0 };
						double best_cbs_cost = { 0.0 };
						double BSBestNNGap = std::numeric_limits<double>::infinity();
						double BSLastNNGap = { 0.0 };
						double CBSBestNNGap = { 0.0 };
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
								if (BSLastNNGap < BSBestNNGap) {
									best_nn_cost = last_nn_cost;
									BSBestNNGap = BSLastNNGap;
								}
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
							CBSBestNNGap = 100 * (best_nn_cost - best_cbs_cost) / best_cbs_cost;
							CBSLastNNGap = 100 * (last_nn_cost - best_cbs_cost) / best_cbs_cost;
							dp.System() << std::endl;
							dp.System() << "------------Uncensored----------LowerCostBetter" << std::endl;
						}
						else {
							CBSBestNNGap = 100 * (best_cbs_cost - best_nn_cost) / best_cbs_cost;
							CBSLastNNGap = 100 * (best_cbs_cost - last_nn_cost) / best_cbs_cost;
							dp.System() << std::endl;
							dp.System() << "------------Censored------------HigherCostBetter" << std::endl;
						}
						dp.System() << "Mean demand: " << demand << "  dist: " << dist << "  p: " << p << "  leadtime: " << leadtime << std::endl;
						dp.System() << "Best base-stock policy cost:  " << best_bs_cost;
						dp.System() << "  last nn_policy_cost:  " << last_nn_cost << "  gaps:  " << BSLastNNGap << "  " << CBSLastNNGap;
						dp.System() << "  best nn_policy_cost:  " << best_nn_cost << "  gaps:  " << BSBestNNGap << "  " << CBSBestNNGap;
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
						results.push_back(best_nn_cost);
						results.push_back(BSBestNNGap);
						results.push_back(CBSBestNNGap);
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

	std::vector<std::string> policy = { "LastNNPolicy", "BestNNPolicy" };
	for (size_t k = 0; k < periods.size(); k++) {
		dp.System() << std::endl;
		dp.System() << "---------Num periods:  " << periods[k] << std::endl;
		dp.System() << std::endl;

		for (size_t j = 0; j < policy.size(); j++) {

			dp.System() << std::endl;
			dp.System() << policy[j] << " results with " << periods[k] << " periods:  " << std::endl;
			dp.System() << std::endl;

			//All results
			PrintResults(AllResults, k, j);
			dp.System() << std::endl;
			dp.System() << std::endl;

			//mean demand results
			for (size_t d = 0; d < mean_demand.size(); d++)
			{
				dp.System() << "Mean demand:  " << mean_demand[d] << "  ";
				PrintResults(meandemandResults[d], k, j);
			}
			dp.System() << std::endl;
			dp.System() << std::endl;

			//distribution results
			for (size_t d = 0; d < dist_token.size(); d++)
			{
				dp.System() << dist_token[d] << " distribution:  " << "  ";
				PrintResults(distResults[d], k, j);
			}
			dp.System() << std::endl;
			dp.System() << std::endl;

			//p values results
			for (size_t d = 0; d < p_values.size(); d++)
			{
				dp.System() << "Penalty cost:  " << p_values[d] << "  ";
				PrintResults(pResults[d], k, j);
			}
			dp.System() << std::endl;
			dp.System() << std::endl;

			//leadtime results
			for (size_t d = 0; d < leadtime_values.size(); d++)
			{
				dp.System() << "Lead time:  " << leadtime_values[d] << "  ";
				PrintResults(leadtimeResults[d], k, j);
			}
			dp.System() << std::endl;
			dp.System() << std::endl;
		}
	}
}

void TrainNetwork() {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",15},
		{"mini_batch_size", 10},
		{"max_training_epochs", 100}
	};

	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
	};

	int64_t num_gens = 5;
	DynaPlex::VarGroup dcl_config{
		//use paper hyperparameters everywhere. 
		{"N",3000000},
		{"num_gens",num_gens},
		{"SimulateOnlyPromisingActions", true},
		{"Num_Promising_Actions", 16},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"retrain_lastgen_only", false}
	};

	//std::string id = "lost_sales_all";
	//std::string id = "lost_sales_all_v2";
	//std::string id = "lost_sales_cyclic";
	std::string id = "lost_sales_all_v3";
	std::string exp_num = "_v66_tsl_tcd_try";
	std::string loc = "dcl_" + id + exp_num;
	dp.System() << "Network id:  " << loc << std::endl;

	bool train = false;
	bool evaluate_paper_instances = false;
	bool evaluate_all_instances = false;
	
	DynaPlex::VarGroup config;
	config.Add("id", "Zero_Shot_Lost_Sales_Inventory_Control");
	config.Add("evaluate", false);
	config.Add("train_stochastic_leadtimes", true);
	config.Add("train_cyclic_demand", true);
	config.Add("train_random_yield", true);
	config.Add("discount_factor", 1.0);
	config.Add("max_demand", 12.0);
	config.Add("max_p", 100.0);
	config.Add("max_leadtime", 10);
	config.Add("max_num_cycles", 7);

	TestAll(config, loc, num_gens);
	if (train) {
		DynaPlex::MDP mdp = dp.GetMDP(config);

		//std::string exp_num_old = "_v21_dummy";
		//std::string loc_old = "dcl_" + id + exp_num_old;
		//auto path = dp.System().filepath(loc_old, "dcl_gen" + 5);
		//auto nn_policy = dp.LoadPolicy(mdp, path);
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

	if (dp.System().WorldRank() == 0 && evaluate_paper_instances)
	{
		TestPolicies(config, loc, num_gens, true, true);
		TestPolicies(config, loc, num_gens, false, true);
	}

	if (dp.System().WorldRank() == 0 && evaluate_all_instances)
	{
		TestPolicies(config, loc, num_gens, false);
		TestPolicies(config, loc, num_gens, true);
	}
}

void TestLeadTime() {

	auto& dp = DynaPlexProvider::Get();

	//std::string id = "lost_sales_all";
	//std::string id = "lost_sales_all_v2";
	//std::string id = "lost_sales_cyclic";
	std::string id = "lost_sales_all_v3";
	std::string exp_num = "_lt_test";
	std::string loc = "dcl_" + id + exp_num;
	dp.System() << "Network id:  " << loc << std::endl;

	DynaPlex::VarGroup config;
	config.Add("id", id);
	config.Add("evaluate", true);
	config.Add("train_stochastic_leadtimes", true);
	config.Add("train_cyclic_demand", false);
	config.Add("train_random_yield", false);
	config.Add("discount_factor", 1.0);
	config.Add("max_demand", 12.0);
	config.Add("max_p", 100.0);
	config.Add("max_leadtime", 10);
	config.Add("max_num_cycles", 7);

	//std::vector<double> lt_probs = { 0.0, 0.0, 0.2, 0.2, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0 };
	std::vector<double> lt_probs = { 0.0, 0.0, 0.2, 0.32, 0.336, 0.144, 0.0, 0.0, 0.0, 0.0, 0.0 };
	config.Add("leadtime_distribution", lt_probs);
	config.Add("order_crossover", true);
	config.Add("mean_demand", 5.0);
	config.Add("stdDemand", std::sqrt(30.0));
	config.Add("p", 39.0);
	config.Add("returnRewards", false);
	config.Add("collectStatistics", false);
	config.Add("censoredDemand", false);
	config.Add("censoredLeadtime", false);

	DynaPlex::MDP mdp = dp.GetMDP(config);
	int64_t BestBSLevel = FindBestBSLevel(config);

}

int main() {
	//TestLeadTime();
	TrainNetwork();

	return 0;
}
