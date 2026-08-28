#include <iostream>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include <cmath>

// Training and testing driver for the Zero_Shot_Lost_Sales_Stationary Super-MDP
// (deterministic lead time, non-cyclic demand). The benchmark search and the policy-comparison
// helpers are borrowed from generally_capable_lost_sales_agent.cpp; the test grid is the "Case 1"
// (stationary) instance set of the original model, which applies directly here. The only
// adaptations are to this model's scalar config schema: mean_demand / stdDemand are scalars (not
// vectors), the lead time is a single deterministic value, and there is no demand_cycles /
// stochastic_leadtime.

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
	// Scalar demand in this model; search the constant-order level up to the mean demand.
	double max_period_demand = 0.0;
	config.Get("mean_demand", max_period_demand);

	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "constant_order");
	policy_config.Add("co_level", COLevel);

	while ((double)COLevel < max_period_demand)
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

		if (innerCBScost < bestCBScost) {
			bestCBScost = innerCBScost;
			bestBSLevel = bs;
			bestCapLevel = innerBestCap;
		}
	}

	return { bestBSLevel, bestCapLevel };
}

// Order-up-to bounds for the benchmark search. Pure DiscreteDist math (independent of the MDP
// config schema); for this model it is called with single-element demand vectors and a
// deterministic (one-point) lead-time distribution.
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

std::vector<std::vector<double>> TestPolicies(DynaPlex::MDP mdp, DynaPlex::MDP test_mdp,
	std::vector<DynaPlex::Policy> policies, DynaPlex::VarGroup instance_config, std::vector<int64_t> periods,
	bool censoredProblem, bool maxReward = false) {

	auto& dp = DynaPlexProvider::Get();
	DynaPlex::Policy test_nn_policy = policies.back();

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("number_of_statistics", 1);
	if (censoredProblem) {
		test_config.Add("warmup_periods", 0);
		policies.pop_back();
	}
	else {
		test_config.Add("warmup_periods", 100);
		periods = { 5000 };
	}

	std::vector<std::vector<double>> AllPeriodResults;
	for (int64_t i = 0; i < periods.size(); i++) {
		int64_t period = periods[i];
		test_config.Set("periods_per_trajectory", period);

		auto comparer = dp.GetPolicyComparer(mdp, test_config);
		auto comparison = comparer.Compare(policies, 0, true, maxReward);

		double last_nn_cost = { 0.0 };
		double best_bs_cost = { 0.0 };
		double best_cbs_cost = { 0.0 };
		double BSLastNNGap = { 0.0 };
		double CBSLastNNGap = { 0.0 };
		double BSCBSGap = { 0.0 };
		double last_nn_service = { 0.0 };
		double best_bs_service = { 0.0 };
		double best_cbs_service = { 0.0 };

		for (auto& VarGroup : comparison)
		{
			DynaPlex::VarGroup policy_id;
			VarGroup.Get("policy", policy_id);
			std::string id;
			policy_id.Get("id", id);

			if (id == "NN_Policy") {
				VarGroup.Get("mean", last_nn_cost);
				VarGroup.Get("mean_gap", BSLastNNGap);
				VarGroup.Get("mean_stat_1", last_nn_service);
			}
			else if (id == "base_stock") {
				VarGroup.Get("mean", best_bs_cost);
				VarGroup.Get("mean_stat_1", best_bs_service);
			}
			else if (id == "capped_base_stock") {
				VarGroup.Get("mean", best_cbs_cost);
				VarGroup.Get("mean_gap", BSCBSGap);
				VarGroup.Get("mean_stat_1", best_cbs_service);
			}
		}

		double test_nn_cost = last_nn_cost;
		double test_nn_gap = { 0.0 };
		double test_nn_service = { 0.0 };
		if (censoredProblem) {
			auto test_comparer = dp.GetPolicyComparer(test_mdp, test_config);
			auto test_comparison = test_comparer.Assess(test_nn_policy);
			test_comparison.Get("mean", test_nn_cost);
			test_comparison.Get("mean_stat_1", test_nn_service);
		}

		if (!maxReward) {
			CBSLastNNGap = 100 * (last_nn_cost - best_cbs_cost) / best_cbs_cost;
			test_nn_gap = 100 * (last_nn_cost - test_nn_cost) / test_nn_cost;
		}
		else {
			CBSLastNNGap = 100 * (best_cbs_cost - last_nn_cost) / best_cbs_cost;
			test_nn_gap = 100 * (test_nn_cost - last_nn_cost) / test_nn_cost;
		}

		std::vector<double> results{};
		results.push_back(best_bs_cost);
		results.push_back(best_cbs_cost);
		results.push_back(BSCBSGap);
		results.push_back(last_nn_cost);
		results.push_back(BSLastNNGap);
		results.push_back(CBSLastNNGap);
		results.push_back(test_nn_gap);
		results.push_back(best_bs_service);
		results.push_back(best_cbs_service);
		results.push_back(last_nn_service);
		results.push_back(test_nn_service);
		AllPeriodResults.push_back(results);
	}

	return AllPeriodResults;
}

void PrintResults(std::vector<std::vector<std::vector<std::vector<double>>>> results, size_t censoredCase, size_t period) {
	auto& dp = DynaPlexProvider::Get();

	double BSCostsAll = 0.0;
	double CBSCostsAll = 0.0;
	double BSCBSGapsAll = 0.0;
	double NNCostsAll = 0.0;
	double BSNNGapsAll = 0.0;
	double CBSNNGapsAll = 0.0;
	double TestNNGapsAll = 0.0;
	double BsServiceAll = 0.0;
	double CbsServiceAll = 0.0;
	double NNServiceAll = 0.0;
	double TestNNService = 0.0;

	for (size_t k = 0; k < results.size(); k++)
	{
		BSCostsAll += results[k][censoredCase][period][0];
		CBSCostsAll += results[k][censoredCase][period][1];
		BSCBSGapsAll += results[k][censoredCase][period][2];
		NNCostsAll += results[k][censoredCase][period][3];
		BSNNGapsAll += results[k][censoredCase][period][4];
		CBSNNGapsAll += results[k][censoredCase][period][5];
		TestNNGapsAll += results[k][censoredCase][period][6];
		BsServiceAll += results[k][censoredCase][period][7];
		CbsServiceAll += results[k][censoredCase][period][8];
		NNServiceAll += results[k][censoredCase][period][9];
		TestNNService += results[k][censoredCase][period][10];
	}
	size_t TotalNumInstanceAll = results.size();

	dp.System() << "Avg BS Costs:  " << BSCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS - NN Policy Gap:  " << BSNNGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg CBS - NN Policy Gap:  " << CBSNNGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg Test NN - NN Policy Gap:  " << TestNNGapsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS Service:  " << BsServiceAll / TotalNumInstanceAll;
	dp.System() << "  , Avg CBS Service:  " << CbsServiceAll / TotalNumInstanceAll;
	dp.System() << "  , Avg NN Service:  " << NNServiceAll / TotalNumInstanceAll;
	dp.System() << "  , Avg Test NN Service:  " << TestNNService / TotalNumInstanceAll;
	dp.System() << "  , Avg NN Policy Costs:  " << NNCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg CBS Policy Costs:  " << CBSCostsAll / TotalNumInstanceAll;
	dp.System() << "  , Avg BS - CBS Gap:  " << BSCBSGapsAll / TotalNumInstanceAll;
}

// Standard deviation implied by a demand distribution family for a given mean (borrowed from the
// original Case 1 setup).
double StdevForDistribution(const std::string& dist, double demand) {
	double p_dummy = 0.3;
	if (dist == "binom") {
		int64_t n = static_cast<int64_t>(std::round(demand / p_dummy));
		double prob = demand / n;
		double var = n * prob * (1 - prob);
		return std::sqrt(var);
	}
	else if (dist == "poisson") {
		return std::sqrt(demand);
	}
	else if (dist == "neg_binom") {
		int64_t r = static_cast<int64_t>(std::ceil(demand * p_dummy / (1 - p_dummy)));
		r = std::max(r, (int64_t)2);
		double prob = (double)r / (demand + r);
		double var = demand / prob;
		return std::sqrt(var);
	}
	else { // geometric
		double prob = 1.0 / (1.0 + demand);
		double var = (1 - prob) / (prob * prob);
		return std::sqrt(var);
	}
}

// "Case 1" (stationary) instance grid of the original model, applied directly to this model.
void TestStationaryInstances(DynaPlex::VarGroup& mdp_config, std::string path, bool testcensored = false) {

	auto& dp = DynaPlexProvider::Get();
	dp.System() << path << std::endl;

	std::vector<double> mean_demand = { 3.0, 5.0, 7.0, 10.0 };
	std::vector<std::string> dist_token = { "binom", "poisson", "neg_binom", "geometric" };
	std::vector<double> p_values = { 9.0, 39.0, 69.0, 99.0 };
	std::vector<int64_t> leadtime_values = { 2, 4, 6, 8, 10 };

	if (testcensored) {
		mean_demand = { 10.0 };
		dist_token = { "poisson", "geometric" };
		p_values = { 5.0, 10.0 };
		leadtime_values = { 1, 3, 5, 7 };
	}
	std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> meandemandResults(mean_demand.size());

	DynaPlex::VarGroup instance_config;
	std::vector<int64_t> periods = { 200, 500, 1000, 2000 };
	std::vector<int64_t> demand_cycles = { 0 }; // for ReturnBounds only (single stationary demand)

	mdp_config.Set("evaluate", true);
	std::vector<std::vector<std::vector<std::vector<double>>>> allResults;
	for (double p : p_values) {
		std::vector<std::vector<std::vector<std::vector<double>>>> Results;
		mdp_config.Set("p", p);
		instance_config.Set("p", p);

		int64_t meandemandIndex = 0;
		for (double demand : mean_demand) {
			mdp_config.Set("mean_demand", demand);   // scalar
			instance_config.Set("mean_demand", demand);

			for (std::string dist : dist_token) {
				double stdev = StdevForDistribution(dist, demand);
				mdp_config.Set("stdDemand", stdev);   // scalar
				instance_config.Set("stdDemand", stdev);

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
					std::vector<double> demand_vec = { demand };
					std::vector<double> stdDemand_vec = { stdev };
					std::pair<int64_t, int64_t> bounds = ReturnBounds(p / (p + 1.0), leadtime_probs, demand_cycles, demand_vec, stdDemand_vec);
					std::pair<int64_t, int64_t> bestParams = FindCBSLevels(mdp_config, BestBSLevel, bounds.second, BestCOLevel, bounds.first);
					int64_t BestSLevel = bestParams.first;
					int64_t BestrLevel = bestParams.second;
					DynaPlex::VarGroup policy_config;
					policy_config.Set("base_stock_level", BestBSLevel);
					policy_config.Set("S", BestSLevel);
					policy_config.Set("r", BestrLevel);

					DynaPlex::VarGroup uncensored_mdp_config = mdp_config;
					uncensored_mdp_config.Set("censoredDemand", false);

					std::vector<bool> censoredDemand_vec = { false, true };
					if (!testcensored) {
						censoredDemand_vec = { false };
					}

					std::vector<std::vector<std::vector<double>>> pResults;
					for (bool censoredDemand : censoredDemand_vec) {
						mdp_config.Set("censoredDemand", censoredDemand);
						instance_config.Set("censoredDemand", censoredDemand);

						DynaPlex::MDP test_mdp = dp.GetMDP(mdp_config);
						std::vector<DynaPlex::Policy> policies;

						policy_config.Set("id", "base_stock");
						policies.push_back(test_mdp->GetPolicy(policy_config));
						policy_config.Set("id", "capped_base_stock");
						policies.push_back(test_mdp->GetPolicy(policy_config));
						policies.push_back(dp.LoadPolicy(test_mdp, path));

						// Under censoring, the network is tested on the true (uncensored) dynamics too.
						DynaPlex::MDP uncensored_test_mdp = dp.GetMDP(uncensored_mdp_config);
						if (censoredDemand)
							policies.push_back(dp.LoadPolicy(uncensored_test_mdp, path));

						pResults.push_back(TestPolicies(test_mdp, uncensored_test_mdp, policies, instance_config, periods, censoredDemand, censoredDemand));
					}
					Results.push_back(pResults);
					allResults.push_back(pResults);
					meandemandResults[meandemandIndex].push_back(pResults);
				}
			}
			meandemandIndex++;
		}
		// Uncensored results

		dp.System() << std::endl;
		dp.System() << "----------------Uncensored Results With Penalty Cost:  " << p << std::endl;
		dp.System() << "---------Num periods:  " << 5000 << std::endl;
		dp.System() << std::endl;

		PrintResults(Results, 0, 0);

		if (testcensored) {
			for (size_t l = 0; l < periods.size(); l++) {
				dp.System() << std::endl;
				dp.System() << "----------------Censored Results With Penalty Cost:  " << p << std::endl;
				dp.System() << "---------Num periods:  " << periods[l] << std::endl;
				dp.System() << std::endl;

				PrintResults(Results, 1, l);
			}
		}
	}

	// Uncensored demand results, split by mean demand

	for (int64_t i = 0; i < meandemandResults.size(); i++) {
		dp.System() << std::endl;
		dp.System() << "----------------Uncensored Results With Mean Demand:  " << mean_demand[i] << std::endl;
		dp.System() << "---------Num periods:  " << 5000 << std::endl;
		dp.System() << std::endl;
		PrintResults(meandemandResults[i], 0, 0);
	}

	// Uncensored results, aggregated

	dp.System() << std::endl;
	dp.System() << "----------------Uncensored Results:  " << std::endl;
	dp.System() << "---------Num periods:  " << 5000 << std::endl;
	dp.System() << std::endl;

	PrintResults(allResults, 0, 0);

	if (testcensored) {
		for (size_t l = 0; l < periods.size(); l++) {
			dp.System() << std::endl;
			dp.System() << "----------------Censored Results:  " << std::endl;
			dp.System() << "---------Num periods:  " << periods[l] << std::endl;
			dp.System() << std::endl;

			PrintResults(allResults, 1, l);
		}
	}
}

void TrainAndTest() {

	auto& dp = DynaPlexProvider::Get();

	// ---- toggles: flip these to train and/or test ---------------------------------------------
	bool train = true;                 // train the GC-LSN-Stationary network via DCL (needs LibTorch)
	bool evaluate_censored_instances = true; // 
	bool evaluate_all_instances = true;  
	// -------------------------------------------------------------------------------------------

	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",15},
		{"mini_batch_size", 1024},
		{"max_training_epochs", 100}
	};

	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
	};

	int64_t num_gens = 5;
	DynaPlex::VarGroup dcl_config{
		{"N",1000000},
		{"num_gens",num_gens},
		{"H",21},
		{"M",500},
		{"L",100},
		{"reinitiate_counter",100},
		{"SimulateOnlyPromisingActions", true},
		{"Num_Promising_Actions", 16},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"retrain_lastgen_only", false}
	};

	DynaPlex::VarGroup config;
	config.Add("id", "Zero_Shot_Lost_Sales_Stationary");
	config.Add("evaluate", false);
	config.Add("discount_factor", 1.0);
	config.Add("max_demand", 12.0);
	config.Add("max_p", 100.0);
	config.Add("max_leadtime", 10);
	auto path = dp.System().filepath("Zero_Shot_Lost_Sales_Stationary", "GC-LSN-Stationary");

	if (train) {
		DynaPlex::MDP mdp = dp.GetMDP(config);
		auto policy = mdp->GetPolicy("greedy_capped_base_stock");
		auto dcl = dp.GetDCL(mdp, policy, dcl_config);
		dcl.TrainPolicy();
		auto last_policy = dcl.GetPolicy(num_gens);
		dp.SavePolicy(last_policy, path);
	}

	if (dp.System().WorldRank() == 0 && evaluate_censored_instances)
	{
		TestStationaryInstances(config, path, true);
	}

	if (dp.System().WorldRank() == 0 && evaluate_all_instances)
	{
		TestStationaryInstances(config, path, false);
	}
}

int main() {

	TrainAndTest();

	return 0;
}