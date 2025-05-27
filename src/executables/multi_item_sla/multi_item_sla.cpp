#include <iostream>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"

using namespace DynaPlex;

class InitialMDPProcessing
{
public:
	VarGroup class_config;

	bool optimalLowerStockings = false;
	int64_t benchmarkAction;
	int64_t greedyAction;
	double penaltyCost;
	int64_t totalActions;
	std::vector<int64_t> targetPolicies;
	std::vector<double> possiblePenaltyCosts;
	std::vector<double> targetPenaltyCosts;
	std::vector<std::vector<double>> possibleVariedPenaltyCosts;
	std::vector<std::vector<double>> targetVariedPenaltyCosts;
	std::vector<double> penaltyCostsTreshold;

private:
	int64_t numberOfItems;
	double aggregateTargetFillRate;
	int64_t leadTime;
	double demand_rate_limit;
	double high_variance_ratio;
	double cost_limit;
	int64_t reviewHorizon;
	int64_t seed;

	double p_min;
	double p_max;
	std::vector<int64_t> lowerBoundPolicy;
	std::vector<int64_t> upperBoundPolicy;
	bool upperBoundPolicyFound = false;

	std::vector<double> holdingCosts;
	std::vector<int64_t> leadTimes;
	std::vector<double> demandRates;
	std::vector<int64_t> highDemandVariance;

	double totalDemandRate;
	std::vector<DiscreteDist> demand_distributions;
	std::vector<DiscreteDist> demand_distributions_over_leadtime;

	std::vector<std::vector<int64_t>> baseStockLevels;
	std::vector<int64_t> concatBaseStockLevels;
	std::vector<int64_t> newConcatBaseStockLevels;
	int64_t newtotalActions;

	std::vector<double> expectedPolicyHoldingCosts;
	std::vector<double> newExpectedPolicyHoldingCosts;
	std::vector<double> expectedPolicyFillRates;
	std::vector<double> newExpectedPolicyFillRates;

public:

	InitialMDPProcessing(VarGroup& config) : class_config{ config }
	{
		auto& dp = DynaPlexProvider::Get();

		class_config.Get("numberOfItems", numberOfItems);
		class_config.Get("aggregateTargetFillRate", aggregateTargetFillRate);
		class_config.Get("leadTime", leadTime);
		class_config.Get("demand_rate_limit", demand_rate_limit);
		class_config.Get("high_variance_ratio", high_variance_ratio);
		class_config.Get("cost_limit", cost_limit);
		class_config.Get("cost_limit", p_min);
		class_config.GetOrDefault("p_max", p_max, 1000000.0);
		class_config.GetOrDefault("seed", seed, 10061994 + 4081965); //lucky birthdays

		DynaPlex::RNG rng(true, seed);

		lowerBoundPolicy.resize(numberOfItems, 0);
		upperBoundPolicy.resize(numberOfItems, 0);
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

		const int64_t numberOfPossibleSlaTargets = 9; // do not change this =  a
		const double incrementalFillRate = 0.005; // do not change this = b
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

		bool targetPolicyFound_exp = false;
		for (int64_t l = 0; l < totalActions; l++) {
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

			// set configs based on expected performance
			if (!targetPolicyFound_exp && expectedFillRate > aggregateTargetFillRate) {
				targetPolicyFound_exp = true;
				class_config.Set("unavoidableCostPerPeriod", expHoldingCost);
				benchmarkAction = l;
				greedyAction = l;
				class_config.Set("benchmarkAction", benchmarkAction);
			}

			dp.System() << "Fr: " << expectedFillRate;
			dp.System() << "  St/o: " << expectedStockouts;
			dp.System() << "  holding costs:  " << expectedPolicyHoldingCosts[l] << "       ";
			for (int64_t stock : stockLevels) {
				dp.System() << "  " << stock;
			}
			dp.System() << std::endl;
		}
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

	void FindBestStaticPolicy (VarGroup& test_config, int64_t reviewHorizonLength, double penalty) {
		auto& dp = DynaPlexProvider::Get();

		SetBenchmarkActionForPolicy(penalty);
		double best_static_afr = newExpectedPolicyFillRates[benchmarkAction];
		double max_target = newExpectedPolicyFillRates[benchmarkAction];
		int64_t best_static = benchmarkAction;
		if (benchmarkAction > 0 && benchmarkAction < totalActions - 1) {
			std::vector<int64_t> lower_subset(newConcatBaseStockLevels.begin() + (benchmarkAction - 1) * numberOfItems, newConcatBaseStockLevels.begin() + benchmarkAction * numberOfItems);
			lowerBoundPolicy = lower_subset;
			max_target = newExpectedPolicyFillRates[benchmarkAction + 1];
		}
		else {
			throw DynaPlex::Error("Search not yet implemented for benchmarkAction = 0 and benchmarkAction = totalActions - 1.");
		}
		
		std::vector<double> changeFR(numberOfItems, 0.0);
		std::vector<double> changeH(numberOfItems, 0.0);
		double aggFillRate = 0.0;
		int64_t num_candidate_policies = 0;
		std::vector<int64_t> concatBSLevels;
		std::vector<int64_t> newConcatBSLevels = newConcatBaseStockLevels;
		while (true) {
			for (size_t i = 0; i < numberOfItems; i++)
			{
				const std::vector<double> statistics = CalculateItemStatistics(i, lowerBoundPolicy[i]);
				aggFillRate += statistics[0];
				changeFR[i] = statistics[1];
				changeH[i] = statistics[2];
			}
			if (aggFillRate / totalDemandRate < max_target)
			{
				double limit = -std::numeric_limits<double>::infinity();
				int64_t bestRatioSKU = 0;
				for (size_t i = 0; i < numberOfItems; i++) {
					double ratio = changeFR[i] / (changeH[i] * holdingCosts[i]);
					if (ratio > limit) {
						bestRatioSKU = i;
						limit = ratio;
					}
				}
				lowerBoundPolicy[bestRatioSKU]++;
				for (int64_t bs_level : lowerBoundPolicy)
					concatBSLevels.push_back(bs_level);
				num_candidate_policies++;
				aggFillRate = 0.0;
			}
			else 
				break;
		}

		class_config.Set("concatBaseStockLevels", concatBSLevels);
		class_config.Set("totalActions", num_candidate_policies);
		TestPerformance(test_config, reviewHorizonLength, penalty, true);
		std::vector<int64_t> best_bs_levels(concatBSLevels.begin() + benchmarkAction * numberOfItems, concatBSLevels.begin() + (benchmarkAction + 1) * numberOfItems);
		aggFillRate = 0.0;
		for (size_t i = 0; i < numberOfItems; i++)
			aggFillRate += CalculateItemStatistics(i, best_bs_levels[i])[0];

		if (aggFillRate / totalDemandRate != best_static_afr) {
			dp.System() << "Best static policy AFR: " << aggFillRate / totalDemandRate << std::endl;
			if (aggFillRate / totalDemandRate < best_static_afr)
				benchmarkAction = best_static;
			else if (aggFillRate / totalDemandRate > best_static_afr)
				benchmarkAction = best_static + 1;

			newConcatBSLevels.insert(newConcatBSLevels.begin() + benchmarkAction * numberOfItems, best_bs_levels.begin(), best_bs_levels.end());
			class_config.Set("concatBaseStockLevels", newConcatBSLevels);
			class_config.Set("totalActions", newtotalActions + 1);
			class_config.Set("benchmarkAction", benchmarkAction);
		}
	}

	void SetGenerator(VarGroup& test_config, int64_t reviewHorizonLength)
	{
		reviewHorizon = reviewHorizonLength;
		class_config.Set("reviewHorizon", reviewHorizon);
		class_config.Set("penaltyCost", 0.0);
		class_config.Set("concatBaseStockLevels", concatBaseStockLevels);
		class_config.Set("totalActions", totalActions);

		auto& dp = DynaPlexProvider::Get();
		DynaPlex::VarGroup policy_config;
		policy_config.Add("id", "base_stock");
		std::vector<double> empricalPolicyFillRates;
		std::vector<double> empricalPolicyExceededStockouts;
		for (int64_t l = 0; l < totalActions; l++)
		{
			class_config.Set("benchmarkAction", l);
			DynaPlex::MDP mdp = dp.GetMDP(class_config);
			auto comparer = dp.GetPolicyComparer(mdp, test_config);
			policy_config.Set("serviceLevelPolicy", l);
			auto policy = mdp->GetPolicy(policy_config);
			auto comparison = comparer.Assess(policy);
			double empricalFillRate;
			comparison.Get("mean_stat_1", empricalFillRate);
			empricalPolicyFillRates.push_back(empricalFillRate);
			double empricalExceededStockouts;
			comparison.Get("mean_stat_3", empricalExceededStockouts);
			empricalPolicyExceededStockouts.push_back(empricalExceededStockouts);
		}

		newConcatBaseStockLevels.clear();
		newExpectedPolicyFillRates.clear();
		newExpectedPolicyHoldingCosts.clear();
		penaltyCostsTreshold.clear();
		possiblePenaltyCosts.clear();
		targetPolicies.clear();
		targetPenaltyCosts.clear();
		possibleVariedPenaltyCosts.clear();
		targetVariedPenaltyCosts.clear();
		double shortfall_amount = totalDemandRate * reviewHorizon;
		double holding_cost = 0.0;
		double p_lower_treshold = p_min;
		bool targetPolicyFound_emp = false;
		int64_t num_policies = 0;
		for (int64_t l = 0; l < totalActions; l++)
		{
			double delta_0 = (empricalPolicyExceededStockouts[l] - shortfall_amount) / (reviewHorizon * (expectedPolicyHoldingCosts[l] - holding_cost));
			double delta_1 = 0.0;
			if (l < totalActions - 1)
				delta_1 = (empricalPolicyExceededStockouts[l + 1] - empricalPolicyExceededStockouts[l]) / (reviewHorizon * (expectedPolicyHoldingCosts[l + 1] - expectedPolicyHoldingCosts[l]));

			if (delta_0 <= delta_1) {
				shortfall_amount = empricalPolicyExceededStockouts[l];
				holding_cost = expectedPolicyHoldingCosts[l];
				double penalty_treshold = p_max;
				if (l < totalActions - 1)
					penalty_treshold = -1 / delta_1;

				std::vector<int64_t> stockLevels = baseStockLevels[l];
				if (p_min <= penalty_treshold) {
					if (penalty_treshold >= p_max) {
						penalty_treshold = p_max;
					}

					for (int64_t bs_level : stockLevels)
						newConcatBaseStockLevels.push_back(bs_level);

					if (!targetPolicyFound_emp && empricalPolicyFillRates[l] > aggregateTargetFillRate) {
						targetPolicyFound_emp = true;
						greedyAction = l;
						benchmarkAction = l;
					}

					newExpectedPolicyFillRates.push_back(expectedPolicyFillRates[l]);
					newExpectedPolicyHoldingCosts.push_back(expectedPolicyHoldingCosts[l]);

					double diff = penalty_treshold - p_lower_treshold;
					double penalty = 0.5 * diff + p_lower_treshold;

					std::vector<double> variedPenaltyCosts(3, 0.0);
					variedPenaltyCosts[0] = 0.25 * diff + p_lower_treshold;
					variedPenaltyCosts[1] = penalty;
					variedPenaltyCosts[2] = 0.75 * diff + p_lower_treshold;

					penaltyCostsTreshold.push_back(penalty_treshold);
					possiblePenaltyCosts.push_back(penalty);
					possibleVariedPenaltyCosts.push_back(variedPenaltyCosts);
					p_lower_treshold = penalty_treshold;
					num_policies++;

					if (penalty_treshold == p_max && l < totalActions - 1) {
						upperBoundPolicy = baseStockLevels[l + 1];
						dp.System() << "Policy: " << l + 1 << " not added - penalty_treshold > p_max" << std::endl;
						break;
					}
				}
				else if (penalty_treshold < p_min) {
					lowerBoundPolicy = stockLevels;
					dp.System() << "Policy: " << l << " not added - penalty_treshold < p_min" << std::endl;
				}
			}
			else {
				dp.System() << "Policy: " << l << " not added - delta_0 > delta_1" << std::endl;
			}
		}

		newtotalActions = num_policies;
		class_config.Set("concatBaseStockLevels", newConcatBaseStockLevels);
		class_config.Set("totalActions", newtotalActions);
		class_config.Set("benchmarkAction", benchmarkAction);
		penaltyCost = possiblePenaltyCosts[benchmarkAction];
		class_config.Set("penaltyCost", penaltyCost);
		targetPolicies.push_back(benchmarkAction);
		targetPenaltyCosts.push_back(penaltyCost);
		targetVariedPenaltyCosts.push_back(possibleVariedPenaltyCosts[benchmarkAction]);
		int64_t policyIncrement = 2;
		for (int64_t i = 1; i <= 3; i++) {
			const int64_t policy = benchmarkAction + i * policyIncrement;
			if (policy < newtotalActions - 1) {
				targetPolicies.push_back(policy);
				targetPenaltyCosts.push_back(possiblePenaltyCosts[policy]);
				targetVariedPenaltyCosts.push_back(possibleVariedPenaltyCosts[policy]);
			}
		}
		PrintImportantMDPParameters(false);
	}

	void TestPerformance(VarGroup& test_config, int64_t reviewHorizonLength, double penalty, bool recordBestPolicy = false)
	{
		reviewHorizon = reviewHorizonLength;
		class_config.Set("reviewHorizon", reviewHorizon);
		penaltyCost = penalty;
		class_config.Set("penaltyCost", penaltyCost);
		int64_t num_actions;
		class_config.Get("totalActions", num_actions);

		auto& dp = DynaPlexProvider::Get();
		double best_cost = std::numeric_limits<double>::infinity();
		double bestPolicy_exceedingStockout = 0.0;
		double bestPolicy_accuracy = 0.0;
		double bestPolicy_fillRate = 0.0;
		
		DynaPlex::VarGroup policy_config;
		policy_config.Add("id", "base_stock");
		for (int64_t l = 0; l < num_actions; l++)
		{ 
			class_config.Set("benchmarkAction", l);
			DynaPlex::MDP mdp = dp.GetMDP(class_config);
			auto comparer = dp.GetPolicyComparer(mdp, test_config);
			policy_config.Set("serviceLevelPolicy", l);
			auto policy = mdp->GetPolicy(policy_config);
			auto comparison = comparer.Assess(policy);
			double cost;
			comparison.Get("mean", cost);
			double empricalFillRate;
			comparison.Get("mean_stat_1", empricalFillRate);
			double empricalStockOuts;
			comparison.Get("mean_stat_2", empricalStockOuts);
			double empricalExceededStockouts;
			comparison.Get("mean_stat_3", empricalExceededStockouts);
			double targetAccuracy;
			comparison.Get("mean_stat_5", targetAccuracy);

			if (cost < best_cost) {
				benchmarkAction = l;
				best_cost = cost;
				bestPolicy_exceedingStockout = empricalExceededStockouts;
				bestPolicy_accuracy = targetAccuracy;
				bestPolicy_fillRate = empricalFillRate;
			}

			if (!recordBestPolicy) {
				dp.System() << l;
				dp.System() << "  Fr: " << empricalFillRate;
				dp.System() << "  St/o: " << empricalStockOuts << " exceeded: " << empricalExceededStockouts << "  " << empricalExceededStockouts / reviewHorizon;
				dp.System() << "  accuracy: " << targetAccuracy;
				dp.System() << "          cost: " << cost;
				dp.System() << std::endl;
			}
		}

		if (recordBestPolicy) {
			dp.System() << "Cost: " << best_cost;
			dp.System() << "   fill rate: " << bestPolicy_fillRate << "  exceeded: " << bestPolicy_exceedingStockout / reviewHorizon;
			dp.System() << "   accuracy: " << bestPolicy_accuracy;
			dp.System() << "   best policy: " << benchmarkAction;
			dp.System() << std::endl;
		}
		class_config.Set("benchmarkAction", benchmarkAction);
		PrintImportantMDPParameters(true);
	}

	void SetPenaltyCostForPolicy(int64_t policy) {
		if (policy > totalActions - 1 || policy < 0) {
			throw DynaPlex::Error("InitialMDPProcessing: target policy should be between 0 and totalActions - 1.");
		}
		else {
			penaltyCost = possiblePenaltyCosts[policy];
			benchmarkAction = policy;
			class_config.Set("benchmarkAction", benchmarkAction);
			class_config.Set("penaltyCost", penaltyCost);
		}
	}

	void SetBenchmarkActionForPolicy(double penalty) {
		if (penaltyCost < p_min || penaltyCost > p_max) 
			throw DynaPlex::Error("InitialMDPProcessing: penaltycost should be between p_min and p_max.");	
		else if (penaltyCost == p_min || penaltyCost == p_max) 
			throw DynaPlex::Error("InitialMDPProcessing: p = p_min or p = p_max not yet implemented.");
		else {
			penaltyCost = penalty;
			class_config.Set("penaltyCost", penaltyCost);
			for (int64_t i = 0; i < totalActions; i++) {
				if (penaltyCost < penaltyCostsTreshold[i]) {
					benchmarkAction = i;
					break;
				}
				if (penaltyCost == penaltyCostsTreshold[i])
					throw DynaPlex::Error("InitialMDPProcessing: p = penaltyCostsTreshold not yet implemented.");
			}
			class_config.Set("benchmarkAction", benchmarkAction);
		}
	}

	void PrintImportantMDPParameters(bool benchmark) {
		auto& dp = DynaPlexProvider::Get();

		if (dp.System().WorldRank() == 0) {
			dp.System() << "FR: " << aggregateTargetFillRate;
			dp.System() << "  horizon: " << reviewHorizon;
			dp.System() << "  n_items: " << numberOfItems;
			dp.System() << "  leadTime: " << leadTime;
			dp.System() << "  var_ratio: " << high_variance_ratio;
			dp.System() << "  demand_rate: " << demand_rate_limit;
			if (benchmark) {
				dp.System() << "  penalty: " << penaltyCost;
				dp.System() << "  policy: " << benchmarkAction;
			}
			else {
				dp.System() << "  penalty: " << possiblePenaltyCosts[benchmarkAction];
				dp.System() << "  policy: " << benchmarkAction;
				dp.System() << "  targets and penalties: ";
				for (int64_t i = 0; i < targetPolicies.size(); i++) {
					const int64_t policy = targetPolicies[i];
					dp.System() << policy << " " << possiblePenaltyCosts[policy] << " ";
				}
				for (int64_t i = 0; i < newtotalActions; i++) {
					dp.System() << i << " " << penaltyCostsTreshold[i] << " ";
				}
			}
			dp.System() << std::endl;
		}
	}
};

static void ExtensiveTrainingTests(InitialMDPProcessing base_pre_mdp, DynaPlex::VarGroup config, DynaPlex::VarGroup test_config, DynaPlex::VarGroup dcl_config) {
	auto& dp = DynaPlexProvider::Get();

	int64_t reviewHorizon;
	config.Get("reviewHorizon", reviewHorizon);
	double penaltyCost;
	config.Get("penaltyCost", penaltyCost);

	base_pre_mdp.TestPerformance(test_config, reviewHorizon, penaltyCost, true);
	int64_t benchmarkAction = base_pre_mdp.benchmarkAction;
	int64_t greedyAction = base_pre_mdp.greedyAction;

	dp.System() << "---------------" << std::endl;
	dp.System() << std::endl;

	DynaPlex::MDP mdp = dp.GetMDP(base_pre_mdp.class_config);
	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "base_stock");
	policy_config.Add("serviceLevelPolicy", benchmarkAction);
	auto best_bs_policy = mdp->GetPolicy(policy_config);
	auto dcl = dp.GetDCL(mdp, best_bs_policy, dcl_config);
	dcl.TrainPolicy();

	if (dp.System().WorldRank() == 0) {
		auto dcl_policies = dcl.GetPolicies();
		policy_config.Set("id", "greedy_dynamic");
		auto dynamic_pol = mdp->GetPolicy(policy_config);
		dcl_policies.push_back(dynamic_pol);
		policy_config.Set("id", "base_stock");
		policy_config.Set("serviceLevelPolicy", greedyAction);
		auto greedy_policy = mdp->GetPolicy(policy_config);
		dcl_policies.push_back(greedy_policy);

		auto comparer = dp.GetPolicyComparer(mdp, test_config);
		auto comparison = comparer.Compare(dcl_policies, 0, true, false);
		for (auto results : comparison) {
			dp.System() << results.Dump() << std::endl;
		}

		dp.System() << std::endl;
		dp.System() << "---------------" << std::endl;
		dp.System() << std::endl;
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

	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);

	int64_t reviewHorizon = 50;
	int64_t numberOfItems = 20;
	double demand_rate_limit = 5.0;
	double high_variance_ratio = 0.5;
	int64_t leadTime = 4;
	double aggregateTargetFillRate = 0.95;

	// start
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

	config.Add("reviewHorizon", reviewHorizon);
	config.Add("aggregateTargetFillRate", aggregateTargetFillRate);
	config.Add("numberOfItems", numberOfItems);
	config.Add("demand_rate_limit", demand_rate_limit);
	config.Add("high_variance_ratio", high_variance_ratio);
	config.Add("leadTime", leadTime);

	InitialMDPProcessing base_pre_mdp(config);
	base_pre_mdp.SetGenerator(test_config, reviewHorizon);
	int64_t benchmarkAction = base_pre_mdp.targetPolicies[1];
	base_pre_mdp.SetPenaltyCostForPolicy(benchmarkAction);
	config.Set("penaltyCost", base_pre_mdp.penaltyCost);
	base_pre_mdp.FindBestStaticPolicy(test_config, reviewHorizon, base_pre_mdp.penaltyCost);
	ExtensiveTrainingTests(base_pre_mdp, config, test_config, dcl_config);
}

static void NewTrain() {

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
	int64_t H_factor = 2;

	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);

	int64_t base_reviewHorizon = 30;
	int64_t numberOfItems = 20;
	double demand_rate_limit = 5.0;
	double high_variance_ratio = 0.5;
	int64_t leadTime = 4;
	double aggregateTargetFillRate = 0.95;

	// start
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
		{"H", H_factor * base_reviewHorizon},
		{"L", 100},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"enable_sequential_halving", true}
	};

	config.Add("reviewHorizon", base_reviewHorizon);
	config.Add("aggregateTargetFillRate", aggregateTargetFillRate);
	config.Add("numberOfItems", numberOfItems);
	config.Add("demand_rate_limit", demand_rate_limit);
	config.Add("high_variance_ratio", high_variance_ratio);
	config.Add("leadTime", leadTime);

	std::vector<int64_t> reviewHorizons = { 10, 30, 50, 70, 90 };
	std::vector<double> targetFillRates = { 0.95, 0.90 };

	for (double fillRate : targetFillRates) {
		config.Set("aggregateTargetFillRate", fillRate);
		InitialMDPProcessing base_pre_mdp(config);

		for (int64_t reviewHorizon : reviewHorizons) {
			config.Set("reviewHorizon", reviewHorizon);
			base_pre_mdp.SetGenerator(test_config, reviewHorizon);
			double penaltyCost = base_pre_mdp.targetPenaltyCosts[1];
			base_pre_mdp.FindBestStaticPolicy(test_config, reviewHorizon, penaltyCost);
			config.Set("penaltyCost", penaltyCost);
			dcl_config.Set("H", H_factor * reviewHorizon);
			//ExtensiveTrainingTests(base_pre_mdp, config, test_config, dcl_config);
		}
	}
}

static void ExtensiveTests() {

	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup config;
	DynaPlex::VarGroup test_config;
	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
	};
	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",15},
		{"max_training_epochs", 100},
		{"train_based_on_probs", false}
	};
	DynaPlex::VarGroup dcl_config{
		{"L", 100},
		{"nn_architecture",nn_architecture},
		{"nn_training",nn_training},
		{"enable_sequential_halving", true}
	};

	// constant parameters
	config.Add("id", "multi_item_sla");
	config.Add("sendBackUnits", false);
	config.Add("cost_limit", 10.0);
	config.Add("backOrderCost", 0.0);
	test_config.Add("warmup_periods", 100);
	test_config.Add("rng_seed", 10061994);
	test_config.Add("number_of_statistics", 8);

	// variable parameters
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);

	int64_t mini_batch = 256;
	int64_t num_generations = 1;
	int64_t N = 100000;
	int64_t M = 1000;
	int64_t H_factor = 2;

	int64_t base_reviewHorizon = 30;
	config.Set("reviewHorizon", base_reviewHorizon);
	int64_t base_numberOfItems = 20;
	config.Set("numberOfItems", base_numberOfItems);
	int64_t base_leadTimes = 4;
	config.Set("leadTime", base_leadTimes);
	double base_demand_rate = 5.0;
	config.Set("demand_rate_limit", base_demand_rate);
	double base_variance_ratio = 0.5;
	config.Set("high_variance_ratio", base_variance_ratio);

	std::vector<double> targetFillRates = { 0.95, 0.90 };
	std::vector<int64_t> reviewHorizons = { 10, 30, 50, 70, 90 };
	std::vector<int64_t> possibleNumberOfItems = { 10, 30 };
	std::vector<int64_t> leadTimes = { 2, 6 };
	std::vector<double> maxDemandRateLimit = { 1.0, 10.0 };
	std::vector<double> highVarianceRatios = { 0.0, 1.0 };

	//start
	nn_training.Set("mini_batch_size", mini_batch);
	dcl_config.Set("nn_training", nn_training);
	dcl_config.Set("N", N);
	dcl_config.Set("M", M);
	dcl_config.Set("H", H_factor * base_reviewHorizon);
	dcl_config.Set("num_gens", num_generations);

	for (double fillRate : targetFillRates) {
		config.Set("aggregateTargetFillRate", fillRate);
		InitialMDPProcessing base_pre_mdp(config);

		base_pre_mdp.SetGenerator(test_config, 90);
		double penaltyCost = base_pre_mdp.targetPenaltyCosts[0];
		config.Set("penaltyCost", penaltyCost);
		for (int64_t reviewHorizon : reviewHorizons) {
			config.Set("reviewHorizon", reviewHorizon);
			dcl_config.Set("H", H_factor * reviewHorizon);
			base_pre_mdp.SetGenerator(test_config, reviewHorizon);
			base_pre_mdp.FindBestStaticPolicy(test_config, reviewHorizon, penaltyCost);
			ExtensiveTrainingTests(base_pre_mdp, config, test_config, dcl_config);
		}
		dcl_config.Set("H", H_factor * base_reviewHorizon);
		config.Set("reviewHorizon", base_reviewHorizon);

		base_pre_mdp.SetGenerator(test_config, base_reviewHorizon);
		std::vector<double> targetPenaltyCosts = base_pre_mdp.targetPenaltyCosts;
		std::vector<double> targetVariedPenaltyCosts = base_pre_mdp.targetVariedPenaltyCosts[1];
		for (double penaltyCost : targetPenaltyCosts) {
			config.Set("penaltyCost", penaltyCost);
			base_pre_mdp.FindBestStaticPolicy(test_config, base_reviewHorizon, penaltyCost);
			ExtensiveTrainingTests(base_pre_mdp, config, test_config, dcl_config);
		}
		for (double penaltyCost : targetVariedPenaltyCosts) {
			config.Set("penaltyCost", penaltyCost);
			base_pre_mdp.FindBestStaticPolicy(test_config, base_reviewHorizon, penaltyCost);
			ExtensiveTrainingTests(base_pre_mdp, config, test_config, dcl_config);
		}
		double base_penaltyCost = targetPenaltyCosts[1];
		config.Set("penaltyCost", base_penaltyCost);

		for (int64_t leadTime : leadTimes) {
			config.Set("leadTime", leadTime);
			InitialMDPProcessing base_pre_mdp(config);
			base_pre_mdp.SetGenerator(test_config, base_reviewHorizon);
			base_pre_mdp.FindBestStaticPolicy(test_config, base_reviewHorizon, base_penaltyCost);
			ExtensiveTrainingTests(base_pre_mdp, config, test_config, dcl_config);
		}
		config.Set("leadTime", base_leadTimes);

		for (double demand_rate_limit : maxDemandRateLimit) {
			config.Set("demand_rate_limit", demand_rate_limit);
			InitialMDPProcessing base_pre_mdp(config);
			base_pre_mdp.SetGenerator(test_config, base_reviewHorizon);
			base_pre_mdp.FindBestStaticPolicy(test_config, base_reviewHorizon, base_penaltyCost);
			ExtensiveTrainingTests(base_pre_mdp, config, test_config, dcl_config);
		}
		config.Set("demand_rate_limit", base_demand_rate);

		for (double high_variance_ratio : highVarianceRatios) {
			config.Set("high_variance_ratio", high_variance_ratio);
			InitialMDPProcessing base_pre_mdp(config);
			base_pre_mdp.SetGenerator(test_config, base_reviewHorizon);
			base_pre_mdp.FindBestStaticPolicy(test_config, base_reviewHorizon, base_penaltyCost);
			ExtensiveTrainingTests(base_pre_mdp, config, test_config, dcl_config);
		}
		config.Set("high_variance_ratio", base_variance_ratio);

		for (int64_t numberOfItems : possibleNumberOfItems) {
			config.Set("numberOfItems", numberOfItems);
			InitialMDPProcessing base_pre_mdp(config);
			base_pre_mdp.SetGenerator(test_config, base_reviewHorizon);
			base_pre_mdp.FindBestStaticPolicy(test_config, base_reviewHorizon, base_penaltyCost);
			ExtensiveTrainingTests(base_pre_mdp, config, test_config, dcl_config);
		}
		config.Set("numberOfItems", base_numberOfItems);
	}
}

static void BaseStockPolicyTests()  {
	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup config;
	DynaPlex::VarGroup test_config;

	// constant parameters
	config.Add("id", "multi_item_sla");
	config.Add("sendBackUnits", false);
	config.Add("cost_limit", 10.0);
	config.Add("backOrderCost", 0.0);
	test_config.Add("warmup_periods", 100);
	test_config.Add("rng_seed", 10061994);
	test_config.Add("number_of_statistics", 8);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);

	std::vector<int64_t> reviewHorizons = { 10, 30, 50, 70, 90 };
	std::vector<double> AFRTargets = { 0.95, 0.90, 0.85 };

	int64_t base_reviewHorizon = 90;
	config.Set("reviewHorizon", base_reviewHorizon);
	config.Set("numberOfItems", 20);
	config.Set("leadTime", 4);
	config.Set("demand_rate_limit", 5.0);
	config.Set("high_variance_ratio", 0.5);

	for (double fillRate : AFRTargets) {
		config.Set("aggregateTargetFillRate", fillRate);

		InitialMDPProcessing base_pre_mdp(config);
		base_pre_mdp.SetGenerator(test_config, base_reviewHorizon);
		std::vector<double> targetPenaltyCosts = base_pre_mdp.targetPenaltyCosts;
		dp.System() << "---------------" << std::endl;

		for (double penalty : targetPenaltyCosts) {
			for (int64_t reviewHorizon : reviewHorizons) {
				base_pre_mdp.TestPerformance(test_config, reviewHorizon, penalty);
				dp.System() << std::endl;
				dp.System() << "---------------" << std::endl;
			}
		}
	}
}

static void BaseStockPolicyTestsRemaining()
{
	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup config;
	DynaPlex::VarGroup test_config;

	// constant parameters
	config.Add("id", "multi_item_sla");
	config.Add("sendBackUnits", false);
	config.Add("cost_limit", 10.0);
	config.Add("backOrderCost", 0.0);
	test_config.Add("warmup_periods", 100);
	test_config.Add("rng_seed", 10061994);
	test_config.Add("number_of_statistics", 8);

	// variable parameters
	std::vector<double> targetFillRates = { 0.95, 0.90};
	std::vector<int64_t> reviewHorizons = { 10, 30, 50, 70, 90 };
	std::vector<int64_t> possibleNumberOfItems = { 10, 20, 30 };
	std::vector<int64_t> leadTimes = { 2, 4, 6 };
	std::vector<double> maxDemandRateLimit = { 1.0, 5.0, 10.0 };
	std::vector<double> highVarianceRatios = { 0.0, 0.5, 1.0 };

	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);

	int64_t base_reviewHorizon = 30;
	config.Set("reviewHorizon", base_reviewHorizon);
	int64_t base_numberOfItems = 20;
	config.Set("numberOfItems", base_numberOfItems);
	int64_t base_leadTimes = 4;
	config.Set("leadTime", base_leadTimes);
	double base_demand_rate = 5.0;
	config.Set("demand_rate_limit", base_demand_rate);
	double base_variance_ratio = 0.5;
	config.Set("high_variance_ratio", base_variance_ratio);

	for (double fillRate : targetFillRates) {
		config.Set("aggregateTargetFillRate", fillRate);

		InitialMDPProcessing base_pre_mdp(config);
		base_pre_mdp.SetGenerator(test_config, base_reviewHorizon);
		double penaltyCost = base_pre_mdp.targetPenaltyCosts[1];
		dp.System() << "---------------" << std::endl;

		for (int64_t leadTime : leadTimes) {
			config.Set("leadTime", leadTime);
			InitialMDPProcessing base_pre_mdp(config);
			for (int64_t reviewHorizon : reviewHorizons) {
				base_pre_mdp.SetGenerator(test_config, reviewHorizon);
				base_pre_mdp.TestPerformance(test_config, reviewHorizon, penaltyCost, true);
				dp.System() << std::endl;
				dp.System() << "---------------" << std::endl;
			}
			dp.System() << std::endl;
			dp.System() << "---------------" << std::endl;
		}
		config.Set("leadTime", base_leadTimes);

		for (double demand_rate_limit : maxDemandRateLimit) {
			config.Set("demand_rate_limit", demand_rate_limit);
			InitialMDPProcessing base_pre_mdp(config);
			for (int64_t reviewHorizon : reviewHorizons) {
				base_pre_mdp.SetGenerator(test_config, reviewHorizon);
				base_pre_mdp.TestPerformance(test_config, reviewHorizon, penaltyCost, true);
				dp.System() << std::endl;
				dp.System() << "---------------" << std::endl;
			}
			dp.System() << std::endl;
			dp.System() << "---------------" << std::endl;
		}
		config.Set("demand_rate_limit", base_demand_rate);

		for (double high_variance_ratio : highVarianceRatios) {
			config.Set("high_variance_ratio", high_variance_ratio);
			InitialMDPProcessing base_pre_mdp(config);
			for (int64_t reviewHorizon : reviewHorizons) {
				base_pre_mdp.SetGenerator(test_config, reviewHorizon);
				base_pre_mdp.TestPerformance(test_config, reviewHorizon, penaltyCost, true);
				dp.System() << std::endl;
				dp.System() << "---------------" << std::endl;
			}
			dp.System() << std::endl;
			dp.System() << "---------------" << std::endl;
		}
		config.Set("high_variance_ratio", base_variance_ratio);

		for (int64_t numberOfItems : possibleNumberOfItems) {
			config.Set("numberOfItems", numberOfItems);
			InitialMDPProcessing base_pre_mdp(config);
			for (int64_t reviewHorizon : reviewHorizons) {
				base_pre_mdp.SetGenerator(test_config, reviewHorizon);
				base_pre_mdp.TestPerformance(test_config, reviewHorizon, penaltyCost, true);
				dp.System() << std::endl;
				dp.System() << "---------------" << std::endl;
			}
			dp.System() << std::endl;
			dp.System() << "---------------" << std::endl;
		}
		config.Set("numberOfItems", base_numberOfItems);
	}
}

int main() {
	//BaseStockPolicyTests();
	//BaseStockPolicyTestsRemaining();
	ExtensiveTests();

	return 0;
}