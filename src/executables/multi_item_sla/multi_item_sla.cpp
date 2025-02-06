#include <iostream>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"

using namespace DynaPlex;

class InitialMDPProcessing
{
public:
	VarGroup class_config;
	bool useEmpiricalPerformance;

	bool optimalLowerStockings = false;
	int64_t benchmarkAction;
	int64_t totalActions;
	int64_t reviewHorizon;
	std::vector<int64_t> targetPolicies;
	std::vector<double> possiblePenaltyCosts;
	std::vector<double> targetPenaltyCosts;

	std::vector<double> expectedPolicyFillRates;
	std::vector<double> expectedPolicyStockouts;
	std::vector<double> empricalPolicyFillRates;
	std::vector<double> empricalPolicyStockouts;
	std::vector<double> empricalPolicyExceededStockouts;

private:
	VarGroup test_config;
	bool printStatistics;

	int64_t numberOfItems;
	double aggregateTargetFillRate;
	int64_t leadTime;
	double demand_rate_limit;
	double high_variance_ratio;
	double cost_limit;

	std::vector<double> holdingCosts;
	std::vector<int64_t> leadTimes;
	std::vector<double> demandRates;
	std::vector<int64_t> highDemandVariance;

	double totalDemandRate;
	std::vector<DiscreteDist> demand_distributions;
	std::vector<DiscreteDist> demand_distributions_over_leadtime;
	std::vector<std::vector<int64_t>> baseStockLevels;

public:

	InitialMDPProcessing(VarGroup& config, VarGroup& test_config, bool useEmpiricalPerformance = false, bool printStatistics = true) :
		class_config{ config }, test_config{ test_config }, useEmpiricalPerformance { useEmpiricalPerformance }, printStatistics{ printStatistics }
	{
		DynaPlex::RNG rng(true, 1923);

		class_config.Get("numberOfItems", numberOfItems);
		class_config.Get("reviewHorizon", reviewHorizon);
		class_config.Get("aggregateTargetFillRate", aggregateTargetFillRate);
		class_config.Get("leadTime", leadTime);
		class_config.Get("demand_rate_limit", demand_rate_limit);
		class_config.Get("high_variance_ratio", high_variance_ratio);
		class_config.Get("cost_limit", cost_limit);

		if (aggregateTargetFillRate >= 1.0 || aggregateTargetFillRate < 0.0)
			throw DynaPlex::Error("InitialMDPProcessing: aggregateTargetFillRate should be between 0 and 1.");

		holdingCosts.reserve(numberOfItems);
		demandRates.reserve(numberOfItems);
		leadTimes.reserve(numberOfItems);
		highDemandVariance.reserve(numberOfItems);
		demand_distributions.reserve(numberOfItems);
		demand_distributions_over_leadtime.reserve(numberOfItems);
		std::vector<int64_t> stockLevels;
		stockLevels.reserve(numberOfItems);
		totalDemandRate = 0.0;
		for (int64_t i = 0; i < numberOfItems; i++)
		{
			holdingCosts.push_back(rng.genUniform() * cost_limit);
			demandRates.push_back(rng.genUniform() * demand_rate_limit);
			leadTimes.push_back(leadTime);
			if (rng.genUniform() > high_variance_ratio) {
				highDemandVariance.push_back(0); 
				demand_distributions.push_back(DiscreteDist::GetPoissonDist(demandRates[i]));
			}
			else {
				highDemandVariance.push_back(1);
				demand_distributions.push_back(DiscreteDist::GetGeometricDist(demandRates[i]));
			}

			if (leadTimes[i] > 0) {
				auto DemOverLeadtime = DiscreteDist::GetZeroDist();
				for (int64_t k = 0; k < leadTimes[i]; k++) {
					DemOverLeadtime = DemOverLeadtime.Add(demand_distributions[i]);
				}
				demand_distributions_over_leadtime.push_back(DemOverLeadtime);
			}
			else {
				demand_distributions_over_leadtime.push_back(DiscreteDist::GetZeroDist());
			}

			totalDemandRate += demandRates[i];
			double rate = demandRates[i] * leadTimes[i] - 1.0;
			if (rate > 0.0) {
				stockLevels.push_back(static_cast<int64_t>(std::ceil(rate)));
			}
			else {
				stockLevels.push_back(0);
			}
		}
		class_config.Set("leadTimes", leadTimes);
		class_config.Set("holdingCosts", holdingCosts);
		class_config.Set("demandRates", demandRates);
		class_config.Set("highDemandVariance", highDemandVariance);
		class_config.Set("totalDemandRate", totalDemandRate);

		const int64_t numberOfPossibleSlaTargets = 9; // do not change this 
		const double incrementalFillRate = 0.005; // do not change this 
		std::vector<int64_t> concatBaseStockLevels;
		for (int64_t l = -numberOfPossibleSlaTargets; l <= numberOfPossibleSlaTargets; l++)
		{
			bool newPolicyFound = false;
			DetermineStockLevels(stockLevels, aggregateTargetFillRate + incrementalFillRate * l, newPolicyFound);
			if (newPolicyFound) {
				baseStockLevels.push_back(stockLevels);
				for (int64_t bs_level : stockLevels)
					concatBaseStockLevels.push_back(bs_level);
			}
		}
		totalActions = baseStockLevels.size();
		class_config.Set("concatBaseStockLevels", concatBaseStockLevels);
		class_config.Set("totalActions", totalActions);

		TestPerformance();
	}

	std::vector<double> CalculateItemStatistics(int64_t item, int64_t stock_level) const
	{
		std::vector<double> statistics(4, 0.0);
		const int64_t maxDemand = demand_distributions[item].Max();
		// expected fill rate, expected fill rate change, expected on hand inventory change, expected on hand inventory, variance of fill rate change
		for (int64_t j = 0; j <= stock_level; j++) {
			const double leadTimeDemandProb = demand_distributions_over_leadtime[item].ProbabilityAt(j);
			double probSums = 0.0;
			const int64_t bound = std::min<int64_t>(stock_level - j, maxDemand);
			for (int64_t k = 0; k <= bound; k++) {
				const double prob = demand_distributions[item].ProbabilityAt(k);
				probSums += prob;
				statistics[0] += leadTimeDemandProb * prob * k;
				statistics[3] += leadTimeDemandProb * prob * (stock_level - j - k);
			}
			const double factor = leadTimeDemandProb * (1.0 - probSums);
			statistics[0] += factor * (stock_level - j);
			statistics[1] += factor;
			statistics[2] += leadTimeDemandProb * probSums;
		}
		return statistics;
	}

	void DetermineStockLevels(std::vector<int64_t>& stockLevels, double serviceLevel, bool& newPolicyFound) const
	{
		double aggFillRate = 0.0;
		std::vector<double> changeFR(numberOfItems, 0.0);
		std::vector<double> changeH(numberOfItems, 0.0);
		for (size_t i = 0; i < numberOfItems; i++)
		{
			const std::vector<double> statistics = CalculateItemStatistics(i, stockLevels[i]);
			aggFillRate += statistics[0];
			changeFR[i] = statistics[1];
			changeH[i] = statistics[2];
		}
		if (aggFillRate / totalDemandRate < serviceLevel)
		{
			newPolicyFound = true;
			double limit = -std::numeric_limits<double>::infinity();
			int64_t bestRatioSKU = 0;
			for (size_t i = 0; i < numberOfItems; i++)
			{
				double ratio = changeFR[i] / (changeH[i] * holdingCosts[i]);
				if (ratio > limit)
				{
					bestRatioSKU = i;
					limit = ratio;
				}
			}
			stockLevels[bestRatioSKU]++;
			DetermineStockLevels(stockLevels, serviceLevel, newPolicyFound);
		}
	}

	void TestPerformance()
	{
		auto& dp = DynaPlexProvider::Get();
		DynaPlex::MDP mdp = dp.GetMDP(class_config);
		auto comparer = dp.GetPolicyComparer(mdp, test_config);
		bool targetPolicyFound_emp = false;
		bool targetPolicyFound_exp = false;
		benchmarkAction = 9;
		std::vector<double> expectedPolicyHoldingCosts;

		DynaPlex::VarGroup policy_config;
		policy_config.Add("id", "base_stock");
		for (int64_t l = 0; l < totalActions; l++) {

			// test emprical performance
			policy_config.Set("serviceLevelPolicy", l);
			auto policy = mdp->GetPolicy(policy_config);
			auto comparison = comparer.Assess(policy);
			double empricalFillRate;
			comparison.Get("mean_stat_1", empricalFillRate);
			empricalPolicyFillRates.push_back(empricalFillRate);
			double empricalStockOuts;
			comparison.Get("mean_stat_2", empricalStockOuts);
			empricalPolicyStockouts.push_back(empricalStockOuts);
			double empricalExceededStockouts;
			comparison.Get("mean_stat_3", empricalExceededStockouts);
			empricalPolicyExceededStockouts.push_back(empricalExceededStockouts);

			// test expected infinite horizon performance
			double expectedFillRate = 0.0;
			double expHoldingCost = 0.0;
			std::vector<int64_t> stockLevels = baseStockLevels[l];
			for (int64_t i = 0; i < numberOfItems; i++)
			{
				const std::vector<double> stats = CalculateItemStatistics(i, stockLevels[i]);
				expectedFillRate += stats[0];
				expHoldingCost += stats[3] * holdingCosts[i];
			}
			expectedFillRate /= totalDemandRate;
			const double expectedStockouts = (1.0 - expectedFillRate) * totalDemandRate;
			expectedPolicyFillRates.push_back(expectedFillRate);
			expectedPolicyHoldingCosts.push_back(expHoldingCost);
			expectedPolicyStockouts.push_back(expectedStockouts);

			// set configs based on empirical performance
			if (!targetPolicyFound_emp && empricalFillRate > aggregateTargetFillRate) {
				targetPolicyFound_emp = true;
				if (useEmpiricalPerformance) {
					benchmarkAction = l;
					class_config.Set("benchmarkAction", benchmarkAction);
					if (expectedFillRate < aggregateTargetFillRate) {
						optimalLowerStockings = true;
						if (printStatistics)
							dp.System() << "Target stocking levels have changed!" << std::endl;
					}
				}
			}

			// set configs based on expected performance
			if (!targetPolicyFound_exp && expectedFillRate > aggregateTargetFillRate) {
				targetPolicyFound_exp = true;
				class_config.Set("unavoidableCostPerPeriod", expHoldingCost);
				if (!useEmpiricalPerformance) {
					benchmarkAction = l;
					class_config.Set("benchmarkAction", benchmarkAction);
				}
			}				

			if (printStatistics) {
				dp.System() << "Fr: " << expectedFillRate << "  " << empricalFillRate;
				dp.System() << "  St/o: " << expectedStockouts * reviewHorizon << "  " << empricalStockOuts << " exceeded: " << empricalExceededStockouts;
				dp.System() << "  holding costs:  " << expHoldingCost << "     ";

				for (int64_t stock : stockLevels) {
					dp.System() << "  " << stock;
				}
				dp.System() << std::endl;
			}
		}

		if (useEmpiricalPerformance && !targetPolicyFound_emp)
			throw DynaPlex::Error("InitialMDPProcessing: benchmark action not found.");

		// set penalty cost tresholds
		for (int64_t i = 0; i < totalActions; i++) {
			double penaltyCost = 0.0;
			if (empricalPolicyExceededStockouts[i] > 1e-3) {
				if (i == 0) {
					const double treshold = (expectedPolicyHoldingCosts[0] - expectedPolicyHoldingCosts[1]) * reviewHorizon
						/ (empricalPolicyExceededStockouts[1] - empricalPolicyExceededStockouts[0]);
					penaltyCost = treshold * 0.5;
				}
				else if (i == totalActions - 1) {
					const double treshold = (expectedPolicyHoldingCosts[totalActions - 1] - expectedPolicyHoldingCosts[totalActions - 2]) * reviewHorizon
						/ (empricalPolicyExceededStockouts[totalActions - 2] - empricalPolicyExceededStockouts[totalActions - 1]);
					penaltyCost = treshold * 1.1;
				}
				else {
					const double treshold_1 = (expectedPolicyHoldingCosts[i] - expectedPolicyHoldingCosts[i - 1]) * reviewHorizon
						/ (empricalPolicyExceededStockouts[i - 1] - empricalPolicyExceededStockouts[i]);
					const double treshold_2 = (expectedPolicyHoldingCosts[i] - expectedPolicyHoldingCosts[i + 1]) * reviewHorizon
						/ (empricalPolicyExceededStockouts[i + 1] - empricalPolicyExceededStockouts[i]);
					penaltyCost = std::min((treshold_2 + treshold_1) / 2.0, treshold_1 * 1.1);
				}
			}
			possiblePenaltyCosts.push_back(penaltyCost);
		}

		targetPolicies.push_back(benchmarkAction);
		int64_t policyIncrement = static_cast<int64_t>(std::floor((double)(totalActions - benchmarkAction + 1) / 3.0));
		for (int64_t i = 1; i <= 2; i++) {
			const int64_t policy = benchmarkAction + i * policyIncrement;
			if (policy < totalActions - 1 && possiblePenaltyCosts[policy] > 0.0) {
				targetPolicies.push_back(policy);
			}
		}
	}

	void SetPenaltyCostForPolicy(int64_t policy) {
		if (policy > totalActions - 1 || policy < 0) {
			throw DynaPlex::Error("InitialMDPProcessing: target policy should be between 0 and totalActions - 1.");
		}
		else {
			class_config.Set("benchmarkAction", policy);
			class_config.Set("penaltyCost", possiblePenaltyCosts[policy]);
		}
	}

	void PrintImportantMDPParameters(int64_t policy = -1)
	{
		if (policy > totalActions - 1 || policy < -1) 
			throw DynaPlex::Error("InitialMDPProcessing: target policy should be between 0 and totalActions - 1.");

		auto& dp = DynaPlexProvider::Get();

		dp.System() << "FR: " << aggregateTargetFillRate;
		dp.System() << "  horizon: " << reviewHorizon;
		dp.System() << "  n_items: " << numberOfItems;
		dp.System() << "  leadTime: " << leadTime;
		dp.System() << "  var_ratio: " << high_variance_ratio;
		dp.System() << "  demand_rate: " << demand_rate_limit;
		if (policy > 0) {
			dp.System() << "  penalty: " << possiblePenaltyCosts[policy];
			dp.System() << "  policy: " << policy;
		}
		else {
			dp.System() << "  penalty: " << possiblePenaltyCosts[benchmarkAction];
			dp.System() << "  policy: " << benchmarkAction;
			dp.System() << "  targets and penalties: ";
			for (int64_t i = 0; i < targetPolicies.size(); i++) {
				const int64_t policy = targetPolicies[i];
				dp.System() << policy << " " << possiblePenaltyCosts[policy] << " ";
			}
		}
		dp.System() << std::endl;
	}
};

static void TestBaseStockPolicies(DynaPlex::VarGroup& mdp_config, DynaPlex::VarGroup& test_config)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(mdp_config);
	auto comparer = dp.GetPolicyComparer(mdp, test_config);
	int64_t totalActions;
	mdp_config.Get("totalActions", totalActions);

	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "base_stock");
	for (int64_t i = 0; i < totalActions; i++) {
		policy_config.Set("serviceLevelPolicy", i);
		auto policy = mdp->GetPolicy(policy_config);
		auto comparison = comparer.Assess(policy);
		dp.System() << comparison.Dump() << std::endl;		
	}
}

static void Train() {

	auto& dp = DynaPlexProvider::Get();
	DynaPlex::VarGroup config;
	DynaPlex::VarGroup test_config;

	// constant parameters
	config.Add("id", "multi_item_sla");
	config.Add("cost_limit", 10.0);
	config.Add("sendBackUnits", false);
	config.Add("backOrderCost", 0.0);
	test_config.Add("warmup_periods", 100);
	test_config.Add("number_of_statistics", 5);
	test_config.Add("rng_seed", 10061994);

	// variable parameters
	int64_t mini_batch = 256;
	int64_t num_generations = 1;
	int64_t N = 100000;
	int64_t M = 1000;
	int64_t H_factor = 1;
	bool train = true;

	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);

	int64_t reviewHorizon = 13;
	int64_t numberOfItems = 10;
	double demand_rate_limit = 10.0;
	double high_variance_ratio = 0.0;
	int64_t leadTime = 4;
	double aggregateTargetFillRate = 0.95;

	// start
	config.Add("reviewHorizon", reviewHorizon);
	config.Add("aggregateTargetFillRate", aggregateTargetFillRate);
	config.Add("numberOfItems", numberOfItems);
	config.Add("demand_rate_limit", demand_rate_limit);
	config.Add("high_variance_ratio", high_variance_ratio);
	config.Add("leadTime", leadTime);

	InitialMDPProcessing pre_mdp(config, test_config, true, true);
	int64_t benchmarkAction = pre_mdp.targetPolicies[0];
	pre_mdp.SetPenaltyCostForPolicy(benchmarkAction);
	pre_mdp.PrintImportantMDPParameters(benchmarkAction);

	if (!train) {
		TestBaseStockPolicies(pre_mdp.class_config, test_config);
	}
	else {
		DynaPlex::VarGroup nn_training{
			{"early_stopping_patience",15},
			{"mini_batch_size", mini_batch},
			{"max_training_epochs", 100}
		};

		DynaPlex::VarGroup nn_architecture{
			{"type","mlp"},
			{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
		};

		DynaPlex::VarGroup dcl_config{
			{"N",N},
			{"num_gens",num_generations},
			{"M",M},
			{"H", H_factor * reviewHorizon},
			{"L", 100},
			{"nn_architecture",nn_architecture},
			{"nn_training",nn_training},
			{"enable_sequential_halving", true}
		};

		DynaPlex::MDP mdp = dp.GetMDP(pre_mdp.class_config);
		DynaPlex::VarGroup policy_config;
		policy_config.Add("id", "base_stock");
		policy_config.Add("serviceLevelPolicy", benchmarkAction);
		auto best_bs_policy = mdp->GetPolicy(policy_config);
		auto dcl = dp.GetDCL(mdp, best_bs_policy, dcl_config);
		dcl.TrainPolicy();

		std::vector<DynaPlex::Policy> policies;
		for (int64_t i = 0; i < pre_mdp.totalActions; i++) {
			policy_config.Set("serviceLevelPolicy", i);
			policies.push_back(mdp->GetPolicy(policy_config));
		}
		policy_config.Set("id", "dynamic");
		if (pre_mdp.useEmpiricalPerformance) {
			policy_config.Set("bestPolicyFillRate", pre_mdp.empricalPolicyFillRates[benchmarkAction]);
			policy_config.Set("policyFillRates", pre_mdp.empricalPolicyFillRates);
		}
		else {
			policy_config.Set("bestPolicyFillRate", pre_mdp.expectedPolicyFillRates[benchmarkAction]);
			policy_config.Set("policyFillRates", pre_mdp.expectedPolicyFillRates);
		}
		auto dynamic_policy = mdp->GetPolicy(policy_config);
		policies.push_back(dynamic_policy);
		auto dcl_policies = dcl.GetPolicies();
		for (auto pol : dcl_policies)
			policies.push_back(pol);

		auto comparer = dp.GetPolicyComparer(mdp, test_config);
		auto comparison = comparer.Compare(policies, benchmarkAction, true, false);
		for (auto results : comparison) {
			dp.System() << results.Dump() << std::endl;
		}
	}
}

static void ExtensiveStaticPolicyTests(DynaPlex::VarGroup config, DynaPlex::VarGroup test_config) {
	auto& dp = DynaPlexProvider::Get();

	InitialMDPProcessing pre_mdp(config, test_config, true, true);
	pre_mdp.PrintImportantMDPParameters();
	int64_t benchmarkAction = pre_mdp.benchmarkAction;
	pre_mdp.SetPenaltyCostForPolicy(benchmarkAction);
	//TestBaseStockPolicies(pre_mdp.class_config, test_config);
	dp.System() << std::endl;
	dp.System() << "---------------" << std::endl;
}

static void ExtensiveTrainingTests(DynaPlex::VarGroup config, DynaPlex::VarGroup test_config, DynaPlex::VarGroup dcl_config) {
	auto& dp = DynaPlexProvider::Get();

	InitialMDPProcessing pre_mdp(config, test_config, true, false);
	std::vector<int64_t> targetPolicies = pre_mdp.targetPolicies;

	dp.System() << "---------------NewMDP++++++++++++++++++++NewMDP----------------" << std::endl;
	for (int64_t i = 0; i < targetPolicies.size(); i++) {
		dp.System() << std::endl;
		dp.System() << std::endl;
		int64_t benchmarkAction = targetPolicies[i];
		pre_mdp.SetPenaltyCostForPolicy(benchmarkAction);
		pre_mdp.PrintImportantMDPParameters(benchmarkAction);

		DynaPlex::MDP mdp = dp.GetMDP(pre_mdp.class_config);
		DynaPlex::VarGroup policy_config;
		policy_config.Add("id", "base_stock");
		policy_config.Add("serviceLevelPolicy", benchmarkAction);
		auto best_bs_policy = mdp->GetPolicy(policy_config);
		auto dcl = dp.GetDCL(mdp, best_bs_policy, dcl_config);
		dcl.TrainPolicy();

		std::vector<DynaPlex::Policy> policies;
		policy_config.Set("id", "dynamic");
		policy_config.Set("bestPolicyFillRate", pre_mdp.empricalPolicyFillRates[benchmarkAction]);
		policy_config.Set("policyFillRates", pre_mdp.empricalPolicyFillRates);
		auto dynamic_policy = mdp->GetPolicy(policy_config);
		policies.push_back(dynamic_policy);
		auto dcl_policies = dcl.GetPolicies();
		for (auto pol : dcl_policies)
			policies.push_back(pol);

		auto comparer = dp.GetPolicyComparer(mdp, test_config);
		auto comparison = comparer.Compare(policies, 1, true, false);
		for (auto results : comparison) {
			dp.System() << results.Dump() << std::endl;
		}
	}

	dp.System() << std::endl;
	dp.System() << "---------------" << std::endl;
	dp.System() << std::endl;
}

static void ExtensiveTests(bool train) {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup config;
	DynaPlex::VarGroup test_config;
	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
	};
	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",15},
		{"mini_batch_size", 256},
		{"max_training_epochs", 100}
	};

	// constant parameters
	config.Add("id", "multi_item_sla");
	config.Add("sendBackUnits", false);
	config.Add("cost_limit", 10.0);
	config.Add("backOrderCost", 0.0);
	test_config.Add("warmup_periods", 100);
	test_config.Add("rng_seed", 10061994);
	test_config.Add("number_of_statistics", 5);

	// variable parameters
	std::vector<double> targetFillRates = { 0.95 };
	std::vector<int64_t> reviewHorizons = { 10, 30, 100 };
	std::vector<int64_t> possibleNumberOfItems = { 10, 50 };
	std::vector<int64_t> leadTimes = { 4 };
	std::vector<double> maxDemandRateLimit = { 10.0, 1.0 };
	std::vector<double> highVarianceRatios = { 0.0, 1.0 };

	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);

	int64_t mini_batch = 256;
	int64_t num_generations = 1;
	int64_t N = 250000;
	int64_t M = 1000;
	int64_t H_factor = 1;

	//start
	nn_training.Set("mini_batch_size", mini_batch);
	DynaPlex::VarGroup dcl_config{
		{"N",N},
		{"num_gens",num_generations},
		{"M",M},
		{"H", H_factor * 100},
		{"L", 100},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"enable_sequential_halving", true}
	};
	for (double fillRate : targetFillRates) {
		config.Set("aggregateTargetFillRate", fillRate);

		for (int64_t reviewHorizon : reviewHorizons) {
			config.Set("reviewHorizon", reviewHorizon);
			dcl_config.Set("H", H_factor * reviewHorizon);

			for (int64_t numberOfItems : possibleNumberOfItems) {
				config.Set("numberOfItems", numberOfItems);

				for (int64_t leadTime : leadTimes) {
					config.Set("leadTime", leadTime);

					for (double demand_rate_limit : maxDemandRateLimit) {
						config.Set("demand_rate_limit", demand_rate_limit);

						for (double high_variance_ratio : highVarianceRatios) {
							config.Set("high_variance_ratio", high_variance_ratio);

							if (train)
								ExtensiveTrainingTests(config, test_config, dcl_config);
							else
								ExtensiveStaticPolicyTests(config, test_config);						
						}
					}
				}
			}
		}
	}
}

int main() {

	Train();
	//ExtensiveTests(true);

	return 0;
}