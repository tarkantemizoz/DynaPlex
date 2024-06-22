#include <iostream>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include <cmath>

using namespace DynaPlex;

int64_t FindBestBSLevel(DynaPlex::VarGroup& config, bool censored = true)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);
	if (censored)
		test_config.Add("warmup_periods", 0);
	else
		test_config.Add("warmup_periods", 100);

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

int64_t FindCOLevel(DynaPlex::VarGroup& config, bool censored = true)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);
	if (censored)
		test_config.Add("warmup_periods", 0);
	else
		test_config.Add("warmup_periods", 100);

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

std::pair<int64_t, int64_t> FindCBSLevels(DynaPlex::VarGroup& config, int64_t min_bs, int64_t max_bs, int64_t min_cap, int64_t max_cap, bool censored = true)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);
	if (censored)
		test_config.Add("warmup_periods", 0);
	else
		test_config.Add("warmup_periods", 100);

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

void TestBSPolicy() {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup config;
	config.Add("id", "lost_sales_censored");
	config.Add("discount_factor", 1.0);
	config.Add("max_demand", 12.0);
	config.Add("min_demand", 2.0);
	config.Add("evaluate", true);
	config.Add("maximization", true);
	config.Add("minimization", false);
	config.Add("collectStatistics", false);
	config.Add("p", 10.0);
	config.Add("leadtime", 7);
	config.Add("mean_demand", 10.0);
	config.Add("stdDemand", std::sqrt(110.0));

	DynaPlex::MDP test_mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 0);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);
	auto comparer = dp.GetPolicyComparer(test_mdp, test_config);

	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "base_stock");
	policy_config.Add("base_stock_level", 85);
	auto best_bs_policy = test_mdp->GetPolicy(policy_config);
	auto comparisonbs = comparer.Assess(best_bs_policy);
	dp.System() << comparisonbs.Dump() << std::endl;

	policy_config.Set("id", "capped_base_stock");
	policy_config.Set("S", 95);
	policy_config.Set("r", 9);
	auto best_cbs_policy = test_mdp->GetPolicy(policy_config);
	auto comparisoncbs = comparer.Assess(best_cbs_policy);
	dp.System() << comparisoncbs.Dump() << std::endl;
}

void TestBSPolicyMix() {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup config;
	config.Add("id", "lost_sales_censored_mix");
	config.Add("discount_factor", 1.0);
	config.Add("max_demand", 12.0);
	config.Add("min_demand", 2.0);
	config.Add("evaluate", true);
	config.Add("returnRewards", true);
	config.Add("censoredDemand", true);
	config.Add("max_p", 100.0);
	config.Add("max_leadtime", 10);
	config.Add("collectStatistics", false);
	config.Add("p", 10.0);
	config.Add("leadtime", 7);
	config.Add("mean_demand", 10.0);
	config.Add("stdDemand", std::sqrt(110.0));

	DynaPlex::MDP test_mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 0);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);
	auto comparer = dp.GetPolicyComparer(test_mdp, test_config);

	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "base_stock");
	policy_config.Add("base_stock_level", 85);
	auto best_bs_policy = test_mdp->GetPolicy(policy_config);
	auto comparisonbs = comparer.Assess(best_bs_policy);
	dp.System() << comparisonbs.Dump() << std::endl;

	policy_config.Set("id", "capped_base_stock");
	policy_config.Set("S", 95);
	policy_config.Set("r", 9);
	auto best_cbs_policy = test_mdp->GetPolicy(policy_config);
	auto comparisoncbs = comparer.Assess(best_cbs_policy);
	dp.System() << comparisoncbs.Dump() << std::endl;
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
	if (censored) {
		config.Set("censoredDemand", true);
		config.Set("collectStatistics", true);
		test_config.Add("warmup_periods", 0);
		if (paperInstances) {
			mean_demand = { 10.0 };
			dist_token = { "poisson", "geometric" };
			if (all) {
				p_values = { 5.0, 10.0 };
				leadtime_values = { 3, 5, 7 };
			}
			else {
				p_values = { penalty };
				leadtime_values = { tau };
			}
		}
	}
	else {
		config.Set("censoredDemand", false);
		config.Set("collectStatistics", false);
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

					int64_t BestBSLevel = FindBestBSLevel(config, censored);
					int64_t BestCOLevel = FindCOLevel(config, censored);
					std::pair<int64_t, int64_t> bestParams = FindCBSLevels(config, BestBSLevel, MaxSystemInv, BestCOLevel, MaxOrderSize, censored);
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
		{"SimulateOnlyPromisingActions", true},
		{"Num_Promising_Actions", 12},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training}
	};

	DynaPlex::VarGroup config;
	config.Add("id", "lost_sales_censored_mix");
	config.Add("discount_factor", 1.0);
	config.Add("max_demand", 12.0);
	config.Add("min_demand", 2.0);
	config.Add("evaluate", false);
	config.Add("returnRewards", false);
	config.Add("censoredDemand", false);
	config.Add("collectStatistics", false);
	config.Add("max_p", 100.0);
	config.Add("max_leadtime", 10);

	//DynaPlex::MDP mdp = dp.GetMDP(config);
	//auto policy = mdp->GetPolicy("greedy_capped_base_stock");
	//auto dcl = dp.GetDCL(mdp, policy, dcl_config);
	//dcl.TrainPolicy();

	std::string loc = "dcl_lost_sales_censored_mix_large_v3";
	//for (size_t gen = 1; gen <= num_gens; gen++)
	//{
	//	auto policy = dcl.GetPolicy(gen);
	//	auto path = dp.System().filepath(loc, "dcl_gen" + gen);
	//	dp.SavePolicy(policy, path);
	//}

	//TestPolicies(config, loc, num_gens, true, true);
	//TestPolicies(config, loc, num_gens, false);
	TestPolicies(config, loc, num_gens, true);
}

void TestNNPolicy() {

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

	int64_t num_gens = 3;
	DynaPlex::VarGroup dcl_config{
		//use paper hyperparameters everywhere. 
		{"N",50000},
		{"num_gens",num_gens},
		{"M",500},
		{"H", 20},
		{"L", 0},
		{"reinitiate_counter", 500},
		{"SimulateOnlyPromisingActions", false},
		{"Num_Promising_Actions", 12},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training}
	};

	//std::vector<double> p_values = { 5.0, 10.0 };
	//std::vector<int64_t> leadtime_values = { 3, 5, 7 };

	std::vector<double> p_values = { 5.0 };
	std::vector<int64_t> leadtime_values = { 7 };

	for (double p : p_values) {

		for (int64_t leadtime : leadtime_values) {

			DynaPlex::VarGroup config;
			config.Add("id", "lost_sales_censored");
			config.Add("discount_factor", 1.0);
			config.Add("p", p);
			config.Add("leadtime", leadtime);
			config.Add("max_demand", 12.0);
			config.Add("min_demand", 2.0);
			config.Add("evaluate", false);
			config.Add("returnRewards", false);
			config.Add("collectStatistics", true);
			config.Add("statTreshold", 100);
			config.Add("updateCycle", 2 * leadtime);

			DynaPlex::MDP mdp = dp.GetMDP(config);
			auto policy = mdp->GetPolicy("greedy_capped_base_stock");
			dp.System() << config.Dump() << std::endl;
			auto dcl = dp.GetDCL(mdp, policy, dcl_config);
			dcl.TrainPolicy();

			std::string part1 = "dcl_lost_sales_censored";
			std::string part2 = "greedyPv15";
			int number1 = static_cast<int>(std::floor(p));
			std::string part3 = "L";
			int number2 = static_cast<int>(leadtime);
			std::string combined = part1 + part2 + std::to_string(number1) + part3 + std::to_string(number2);

			for (size_t gen = 1; gen <= num_gens; gen++)
			{
				auto policy = dcl.GetPolicy(gen);
				auto path = dp.System().filepath(combined, "dcl_gen" + gen);
				dp.SavePolicy(policy, path);
			}

			TestPolicies(config, combined, num_gens, true, true, false, p, leadtime);
		}
	}
}

int main() {

	//DynaPlex::DiscreteDist dist = DiscreteDist::GetAdanEenigeResingDist(12.0, 12.0 * 2);
	////Initiate members that are computed from the parameters:
	//auto DemOverLeadtime = DiscreteDist::GetZeroDist();
	//for (size_t i = 0; i <= 7; i++)
	//{
	//	DemOverLeadtime = DemOverLeadtime.Add(dist);
	//}
	//double frac = 5 / (5 + 1.0) * 0.9;
	//int64_t MaxOrderSize = dist.Fractile(frac);
	//int64_t MaxSystemInv = DemOverLeadtime.Fractile(frac);
	//std::cout << frac << "  " <<   MaxOrderSize << "  " << MaxSystemInv << std::endl;

	//TestBSPolicy();
	TrainNetwork();
	//TestNNPolicy();
	//TestBSPolicyMix();
	//TestBSPolicyMix();

	//double min_demand = 2.0;
	//double max_demand = 12.0;
	//double demand = min_demand;

	//std::vector<std::vector<int64_t>> ordervector;
	//std::vector<std::vector<int64_t>> invvector;

	//std::vector<double> demand_vector;
	//std::vector<std::vector<double>> std_vector;

	//while (demand <= max_demand) {
	//	demand_vector.push_back(demand);
	//	double max_std = demand * 2;
	//	double p_dummy = 0.2;
	//	int64_t n = static_cast<int64_t>(std::round(demand / p_dummy));
	//	double prob = demand / n;
	//	double var = n * prob * (1 - prob);
	//	double stdev = ceil(std::sqrt(var));
	//	std::vector<int64_t> max_orders;
	//	std::vector<int64_t> max_invs;
	//	std::vector<double> stds;
	//	while (stdev <= max_std) {
	//		stds.push_back(stdev);

	//		DynaPlex::DiscreteDist demand_dist = DiscreteDist::GetAdanEenigeResingDist(demand, stdev);

	//		auto DemOverLeadtime = DiscreteDist::GetZeroDist();
	//		for (size_t i = 0; i <= 7; i++)
	//		{
	//			DemOverLeadtime = DemOverLeadtime.Add(demand_dist);
	//		}
	//		int64_t MaxOrderSize = demand_dist.Fractile(10.0 / (10.0 + 1.0));
	//		int64_t MaxSystemInv = DemOverLeadtime.Fractile(10.0 / (10.0 + 1.0));
	//		std::cout << "-----demand:  " << demand << "  stdev:  " << stdev << "  " << MaxOrderSize << "  " << MaxSystemInv << std::endl;
	//		max_orders.push_back(MaxOrderSize);
	//		max_invs.push_back(MaxSystemInv);
	//		stdev += 1.0;
	//	}
	//	std_vector.push_back(stds);
	//	ordervector.push_back(max_orders);
	//	invvector.push_back(max_invs);
	//	demand += 1.0;
	//}

	//for (size_t i = 0; i < demand_vector.size(); i++)
	//{
	//	std::cout << i << "  " << demand_vector[i] << std::endl;
	//}
	//for (size_t i = 0; i < std_vector.size(); i++)
	//{
	//	std::cout << i << "  " << std_vector[1][i] << std::endl;
	//}
	//double std = 4.2;
	//demand = 1.8;
	//int64_t ind = findGreaterIndex(demand_vector, demand);
	//std::cout << "Loc:  " << ind << "  " << findGreaterIndex(std_vector[ind], std) << "  " << ordervector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << "  " << invvector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << std::endl;
	//demand = 3.8;
	//std::cout << "Loc:  " << ind << "  " << findGreaterIndex(std_vector[ind], std) << "  " << ordervector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << "  " << invvector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << std::endl;
	//demand = 8.8;
	//std::cout << "Loc:  " << ind << "  " << findGreaterIndex(std_vector[ind], std) << "  " << ordervector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << "  " << invvector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << std::endl;
	//demand = 11.8;
	//std::cout << "Loc:  " << ind << "  " << findGreaterIndex(std_vector[ind], std) << "  " << ordervector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << "  " << invvector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << std::endl;
	//demand = 12.8;
	//std::cout << "Loc:  " << ind << "  " << findGreaterIndex(std_vector[ind], std) << "  " << ordervector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << "  " << invvector[findGreaterIndex(demand_vector, demand)][findGreaterIndex(std_vector[ind], std)] << std::endl;

	return 0;
}
