#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"

namespace DynaPlex::Models {
	namespace multi_item_sla /*keep this in line with id below and with namespace name in header*/
	{
		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;		
			vars.Add("valid_actions", totalActions);
			
			return vars;
		}

		MDP::MDP(const VarGroup& config)
		{
			config.Get("aggregateTargetFillRate", aggregateTargetFillRate);
			config.Get("numberOfItems", numberOfItems);
			config.Get("reviewHorizon", reviewHorizon);
			config.Get("leadTimes", leadTimes);
			config.Get("holdingCosts", holdingCosts);
			config.Get("demandRates", demandRates);
			config.Get("highDemandVariance", highDemandVariance);
			config.Get("totalDemandRate", totalDemandRate);
			config.Get("totalActions", totalActions);
			config.GetOrDefault("unavoidableCostPerPeriod", unavoidableCostPerPeriod, 0.0);
			config.GetOrDefault("benchmarkAction", benchmarkAction, static_cast<int64_t>(std::floor((double)totalActions / 2.0)));
			config.GetOrDefault("penaltyCost", penaltyCost, 100.0);
			config.Get("sendBackUnits", sendBackUnits);
			config.Get("backOrderCost", backOrderCost);

			totalActions++;
			sendBackCost = 0.0;

			demand_distributions.reserve(numberOfItems);
			for (int64_t i = 0; i < numberOfItems; i++) {
				if (highDemandVariance[i] == 1)
					demand_distributions.push_back(DynaPlex::DiscreteDist::GetGeometricDist(demandRates[i]));
				else 
					demand_distributions.push_back(DynaPlex::DiscreteDist::GetPoissonDist(demandRates[i]));
			}

			baseStockLevels.reserve(totalActions - 1);
			std::vector<int64_t> concatBaseStockLevels;
			config.Get("concatBaseStockLevels", concatBaseStockLevels);
			for (int64_t l = 0; l < totalActions - 1; l++)
			{
				std::vector<int64_t> subset(concatBaseStockLevels.begin() + l * numberOfItems, concatBaseStockLevels.begin() + (l + 1) * numberOfItems);
				baseStockLevels.push_back(subset);
			}
			//std::vector<std::vector<int64_t>> itemLevels(numberOfItems, std::vector<int64_t>(totalActions - 1, 0));
			//for (size_t action = 0; action < totalActions - 1; ++action) {
			//	for (size_t item = 0; item < numberOfItems; ++item) {
			//		itemLevels[item][action] = baseStockLevels[action][item];
			//	}
			//}
			//itemStockLevels = itemLevels;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const
		{
			return state.AllowedActions[action];
		}

		void MDP::SetAllowedActions(State& state) const
		{			
			state.AllowedActions.resize(totalActions, false);
			state.AllowedActions[0] = true;
			for (int64_t j = 1; j < totalActions - 1; ++j)
			{
				const auto& bs_levels = baseStockLevels[j - 1];
				const auto& bs_levels_high = baseStockLevels[j];
				for (int64_t i = 0; i < numberOfItems; ++i)
				{
					if (bs_levels[i] > state.inventory_position[i] && bs_levels[i] < bs_levels_high[i]) {
						state.AllowedActions[j] = true;
						break;
					}
				}
			}

			const auto& bs_levels = baseStockLevels[totalActions - 2];
			for (int64_t i = 0; i < numberOfItems; ++i)
			{
				if (bs_levels[i] > state.inventory_position[i]) {
					state.AllowedActions[totalActions - 1] = true;
					break;
				}
			}
		}

		double MDP::ModifyStateWithAction(State& state, int64_t action) const
		{
			double cost = 0.0;	
			
			if (action > 0) {
				const std::vector<int64_t> New_BS_Levels = baseStockLevels[action - 1];
				for (int64_t i = 0; i < numberOfItems; i++)
				{
					const int64_t toOrder = New_BS_Levels[i] - state.inventory_position[i];
					if (toOrder >= 0) {
						if (leadTimes[i] == 0)
							state.state_vector[i].front() += toOrder;
						else
							state.state_vector[i].push_back(toOrder);
						state.inventory_position[i] += toOrder;
					}
					else {
						if (leadTimes[i] != 0)
							state.state_vector[i].push_back(0);
						if (sendBackUnits && state.state_vector[i].front() > 0) {
							int64_t numSendBackUnits = std::min(-toOrder, state.state_vector[i].front());
							state.state_vector[i].front() -= numSendBackUnits;
							state.inventory_position[i] -= numSendBackUnits;
							cost += sendBackCost * numSendBackUnits;
						}
					}
				}
			}
			else {
				for (int64_t i = 0; i < numberOfItems; i++)
				{
					if (leadTimes[i] == 0)
						state.state_vector[i].front() += 0;
					else
						state.state_vector[i].push_back(0);
				}
			}

			if (action != state.LastPolicy) {
				state.LastPolicy = action;
				state.ChangeInPolicy++;
			}

			state.cat = StateCategory::AwaitEvent();

			return cost;
		}

		MDP::Event MDP::GetEvent(RNG& rng) const 
		{
			std::vector<int64_t> events;
			events.reserve(numberOfItems);
			for (int64_t i = 0; i < numberOfItems; i++)
			{
				events.push_back(demand_distributions[i].GetSample(rng));
			}

			return events;
		}

		double MDP::ModifyStateWithEvent(State& state, const MDP::Event& event) const
		{
			state.cat = StateCategory::AwaitAction();
			double cost = 0.0;
			state.TimeRemaining--;
			state.aggregate_vector.clear();

			for (int64_t i = 0; i < numberOfItems; i++)
			{
				auto& currentState = state.state_vector[i];
				int64_t onHand = currentState.pop_front();
				const int64_t demand = event[i];
				int64_t& invPos = state.inventory_position[i];
				invPos -= demand;
				state.ObservedDemand += demand;

				if (onHand >= demand)
				{
					onHand -= demand;
					cost += onHand * holdingCosts[i];
				}
				else
				{
					const int64_t numStockout = demand - (onHand > 0 ? onHand : 0);
					state.CumulativeStockouts += numStockout;
					cost += backOrderCost * numStockout;
					onHand -= demand; 
				}

				if (leadTimes[i] == 0)
					currentState.push_back(onHand);
				else
					currentState.front() += onHand;

				state.aggregate_vector.insert(state.aggregate_vector.end(), currentState.begin(), currentState.end());

				//bool actionFound = false;
				//const auto& bs_levels = itemStockLevels[i];
				//for (int64_t j = 0; j < totalActions - 2; j++) {
				//	if (invPos < bs_levels[j] && bs_levels[j] < bs_levels[j + 1]) {
				//		//counts[j]++;
				//		//actionFound = true;					
				//		allowedActions[j + 1] = true;
				//		break;
				//	}
				//}
				//if (invPos < bs_levels[totalActions - 2]) {
				//	allowedActions[totalActions - 1] = true;
				//	//if (!actionFound)
				//	//	counts[totalActions - 2]++;
				//}
				//for (auto inv : state.state_vector[i]) {
				//	state.aggregate_vector.push_back(inv);
				//}
			}
			//state.ActionStats = std::move(counts);
			//state.HoldingCosts += cost;

			if (state.ObservedDemand > 0)
				state.AggregateFillRate = static_cast<double>(state.ObservedDemand - state.CumulativeStockouts) / (static_cast<double>(state.ObservedDemand));
			
			if (state.TimeRemaining == 0)
			{
				state.NumReviewPeriodPassed++;
				const int64_t shouldSatisfy = static_cast<int64_t>(std::ceil(aggregateTargetFillRate * state.ObservedDemand));
				const int64_t maxStockoutsAllowed = state.ObservedDemand - shouldSatisfy;
				int64_t exceededStockouts = state.CumulativeStockouts - maxStockoutsAllowed;
				if (exceededStockouts > 0)
				{
					cost += exceededStockouts * penaltyCost;
					state.CESPerReviewPeriod = (state.CESPerReviewPeriod * (state.NumReviewPeriodPassed - 1) + (double)exceededStockouts) / (double)state.NumReviewPeriodPassed;
					state.SPPerReviewPeriod = (state.SPPerReviewPeriod * (state.NumReviewPeriodPassed - 1) + 0.0) / (double)state.NumReviewPeriodPassed;
				}
				else {
					state.CESPerReviewPeriod = (state.CESPerReviewPeriod * (state.NumReviewPeriodPassed - 1) + 0.0) / (double)state.NumReviewPeriodPassed;
					state.SPPerReviewPeriod = (state.SPPerReviewPeriod * (state.NumReviewPeriodPassed - 1) + 1.0) / (double)state.NumReviewPeriodPassed;
				}
				state.AFRPerReviewPeriod = (state.AFRPerReviewPeriod * (state.NumReviewPeriodPassed - 1) + state.AggregateFillRate) / (double)state.NumReviewPeriodPassed;
				state.CSPerReviewPeriod = (state.CSPerReviewPeriod * (state.NumReviewPeriodPassed - 1) + (double)state.CumulativeStockouts) / (double)state.NumReviewPeriodPassed;
				state.CIPPerReviewPeriod = (state.CIPPerReviewPeriod * (state.NumReviewPeriodPassed - 1) + (double)state.ChangeInPolicy) / (double)state.NumReviewPeriodPassed;
				//state.HoldingCostsPerReviewPeriod = (state.HoldingCostsPerReviewPeriod * (state.NumReviewPeriodPassed - 1) + state.HoldingCosts) / (double)state.NumReviewPeriodPassed;
				state.ChangeInPolicy = 0;
				state.ObservedDemand = 0;
				state.AggregateFillRate = 1.0;
				state.TimeRemaining = reviewHorizon;
				state.CumulativeStockouts = 0;
			}

			SetAllowedActions(state);
			return cost - unavoidableCostPerPeriod;
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			//features.Add(state.ActionStats);
			features.Add(state.aggregate_vector);
			if (state.TimeRemaining < reviewHorizon) {
				features.Add(state.AggregateFillRate);
				features.Add(static_cast<float>(state.CumulativeStockouts) / static_cast<float>(totalDemandRate * (reviewHorizon - state.TimeRemaining)));
				features.Add(static_cast<float>(state.ObservedDemand) / static_cast<float>(totalDemandRate * (reviewHorizon - state.TimeRemaining)));
			}
			else {
				features.Add(1.0);
				features.Add(0.0);
				features.Add(0.0);
			}
			features.Add(static_cast<float>(state.TimeRemaining) / static_cast<float>(reviewHorizon));
		}

		std::vector<double> MDP::ReturnUsefulStatistics(const State& state) const
		{
			std::vector<double> statistics;
			statistics.reserve(5);
			statistics.push_back(state.AFRPerReviewPeriod);
			statistics.push_back(state.CSPerReviewPeriod);
			statistics.push_back(state.CESPerReviewPeriod);
			statistics.push_back(state.CIPPerReviewPeriod);
			statistics.push_back(state.SPPerReviewPeriod);
			//statistics.push_back(state.HoldingCosts);
			return statistics;
		}

		void MDP::ResetHiddenStateVariables(State& state, RNG& rng) const
		{
			state.ObservedDemand = 0;
			state.AggregateFillRate = 1.0;
			state.TimeRemaining = reviewHorizon;
			state.CumulativeStockouts = 0;
			state.ChangeInPolicy = 0;
			state.NumReviewPeriodPassed = 0;
			//state.HoldingCosts = 0.0;
		} 
		
		MDP::State MDP::GetInitialState() const
		{
			State state{};
			state.cat = StateCategory::AwaitAction();
			state.state_vector.reserve(numberOfItems);
			state.inventory_position.reserve(numberOfItems);
			int64_t aggregateVectorLength = 0;
			for (int64_t i = 0; i < numberOfItems; i++) {
				auto queue = Queue<int64_t>{}; 
				queue.reserve(leadTimes[i] + 1);
				queue.push_back(baseStockLevels[benchmarkAction][i]);//<- initial on-hand
				for (int64_t j = 0; j < leadTimes[i] - 1; j++)
				{
					queue.push_back(0);
				}
				state.state_vector.push_back(queue);
				state.inventory_position.push_back(baseStockLevels[benchmarkAction][i]);
				aggregateVectorLength += leadTimes[i] == 0 ? 1 : leadTimes[i];
			}
			state.aggregate_vector.reserve(aggregateVectorLength);
			for (int64_t i = 0; i < numberOfItems; i++) {
				state.aggregate_vector.insert(state.aggregate_vector.end(), state.state_vector[i].begin(), state.state_vector[i].end());
			}

			state.LastPolicy = benchmarkAction;
			state.ObservedDemand = 0;
			state.AggregateFillRate = 1.0;
			state.TimeRemaining = reviewHorizon;
			state.CumulativeStockouts = 0;
			state.ChangeInPolicy = 0;
			state.NumReviewPeriodPassed = 0;
			state.AFRPerReviewPeriod = 1.0;
			state.CSPerReviewPeriod = 0.0;
			state.CESPerReviewPeriod = 0.0;
			state.CIPPerReviewPeriod = 0.0;
			state.SPPerReviewPeriod = 0.0;
			state.AllowedActions.resize(totalActions, false);
			state.AllowedActions[0] = true;
			//state.HoldingCosts = 0.0;
			//std::vector<int64_t> counts(totalActions - 1, 0);
			//state.ActionStats = std::move(counts);

			return state;
		}
	
		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			//vars.Get("ActionStats", state.ActionStats);
			vars.Get("aggregate_vector", state.aggregate_vector);
			vars.Get("ObservedDemand", state.ObservedDemand);
			vars.Get("AggregateFillRate", state.AggregateFillRate);
			vars.Get("TimeRemaining", state.TimeRemaining);
			vars.Get("CumulativeStockouts", state.CumulativeStockouts);
			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			//vars.Add("ActionStats", ActionStats);
			vars.Add("aggregate_vector", aggregate_vector);
			vars.Add("ObservedDemand", ObservedDemand);
			vars.Add("AggregateFillRate", AggregateFillRate);
			vars.Add("TimeRemaining", TimeRemaining);
			vars.Add("CumulativeStockouts", CumulativeStockouts);
			return vars;
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}
		
		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{
			registry.Register<BaseStockPolicy>("base_stock",
				"Base-stock policy with parameter base_stock_level for all SKUs.");
			registry.Register<DynamicPolicy>("dynamic",
				"Dynamic base-stock policy for all SKUs.");
		}
		
		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "multi_item_sla",
				/*description*/ "Single location multi item inventory management under a service level agreement.",
				/*reference to passed registry*/registry); 
		}
	}
}

//void StockOutCalculation{
//				for (int64_t l = 0; l < baseStockLevels.size(); l++)
//	{
//		double aggFillRate = 0.0;
//		double expHoldingCost = 0.0;
//		stockLevels = baseStockLevels[l];
//		//DynaPlex::DiscreteDist totalStockoutDist = DiscreteDist::GetZeroDist();
//		//DynaPlex::DiscreteDist totalStockoutOverHorizonDist = DiscreteDist::GetZeroDist();
//		//std::vector<std::vector<double>> totalDemandStockoutProbs(1);
//		//totalDemandStockoutProbs[0].push_back(1.0);

//		for (int64_t i = 0; i < numberOfItems; i++)
//		{
//			//const int64_t maxDemand = demand_distributions[i].Max();
//			//const int64_t maxLeadTimeDemand = demand_distributions_over_leadtime[i].Max();
//			const int64_t stockLevel = stockLevels[i];

//			//std::vector<double> stockoutProbs(maxDemand + 1, 0.0);
//			//std::vector<std::vector<double>> demandStockoutProbs(maxDemand + 1);
//			//for (int64_t r = 0; r <= maxDemand; r++) {
//			//	demandStockoutProbs[r].resize(maxDemand - r + 1, 0.0);
//			//}

//			std::vector<double> stats = CalculateItemStatistics(i, stockLevel);
//			aggFillRate += stats[0];
//			expHoldingCost += stats[4] * holdingCosts[i];
//			//for (int64_t j = 0; j <= stockLevel; j++) {
//			//	double leadTimeDemandProb = demand_distributions_over_leadtime[i].ProbabilityAt(j);
//			//	const int64_t bound = std::min<int64_t>(stockLevel - j, maxDemand);
//			//	for (int64_t k = 0; k <= bound; k++) {
//			//		stockoutProbs[0] += leadTimeDemandProb * demand_distributions[i].ProbabilityAt(k);
//			//		demandStockoutProbs[0][k] += leadTimeDemandProb * demand_distributions[i].ProbabilityAt(k);
//			//	}
//			//	for (int64_t k = stockLevel - j + 1; k <= maxDemand; k++) {
//			//		stockoutProbs[k - stockLevel + j] += leadTimeDemandProb * demand_distributions[i].ProbabilityAt(k);
//			//		demandStockoutProbs[k - stockLevel + j][stockLevel - j] += leadTimeDemandProb * demand_distributions[i].ProbabilityAt(k);
//			//	}
//			//}
//			//for (int64_t j = stockLevel + 1; j <= maxLeadTimeDemand; j++) {
//			//	double leadTimeDemandProb = demand_distributions_over_leadtime[i].ProbabilityAt(j);
//			//	for (int64_t k = 0; k <= maxDemand; k++) {
//			//		stockoutProbs[k] += leadTimeDemandProb * demand_distributions[i].ProbabilityAt(k);
//			//		demandStockoutProbs[k][0] += leadTimeDemandProb * demand_distributions[i].ProbabilityAt(k);
//			//	}
//			//}
//			//DynaPlex::DiscreteDist StockoutDist = DiscreteDist::GetCustomDist(stockoutProbs, 0);
//			//DynaPlex::DiscreteDist StockoutDistOverHorizon = DiscreteDist::GetZeroDist();
//			//for (int64_t j = 0; j < reviewHorizon; j++) {
//			//	StockoutDistOverHorizon = StockoutDistOverHorizon.Add(StockoutDist);
//			//}
//			//totalStockoutDist = totalStockoutDist.Add(StockoutDist);
//			//totalStockoutOverHorizonDist = totalStockoutOverHorizonDist.Add(StockoutDistOverHorizon);

//			//std::vector<std::vector<double>> oldTotal = std::move(totalDemandStockoutProbs);
//			//const int64_t oldRows = (int64_t)oldTotal.size();        
//			//const int64_t newRows = (int64_t)demandStockoutProbs.size();
//			//const int64_t resultMaxRow = oldRows + newRows - 2; 
//			//std::vector<std::vector<double>> newTotalDemandStockoutProbs(resultMaxRow + 1);
//			//for (int64_t r = 0; r <= resultMaxRow; r++) {
//			//	newTotalDemandStockoutProbs[r].resize(resultMaxRow - r + 1, 0.0);
//			//}

//			//for (int64_t r1 = 0; r1 < oldRows; r1++) {
//			//	for (int64_t c1 = 0; c1 < (int64_t)oldTotal[r1].size(); c1++) {
//			//		double val1 = oldTotal[r1][c1];
//			//		if (val1 == 0.0) 
//			//			continue;
//			//		for (int64_t r2 = 0; r2 < newRows; r2++) {
//			//			for (int64_t c2 = 0; c2 < (int64_t)demandStockoutProbs[r2].size(); c2++)
//			//				newTotalDemandStockoutProbs[r1 + r2][c1 + c2] += val1 * demandStockoutProbs[r2][c2];
//			//		}
//			//	}
//			//}
//			//totalDemandStockoutProbs = std::move(newTotalDemandStockoutProbs);
//		}

//		//double expExceededStockout = 0.0;
//		//const int64_t maxStockOut = totalDemandStockoutProbs.size() - 1;
//		//for (int64_t s = 0; s <= maxStockOut; s++) {
//		//	const std::vector<double> stockoutVec = totalDemandStockoutProbs[s];
//		//	const int64_t demand = stockoutVec.size() - 1;
//		//	for (int64_t d = 0; d <= demand; d++) {
//		//		const int64_t shouldSatisfy = static_cast<int64_t>(std::ceil(aggregateTargetFillRate * (d + s)));
//		//		const int64_t allowableStockouts = (d + s) - shouldSatisfy;
//		//		if (s > allowableStockouts) {
//		//			expExceededStockout += (s - allowableStockouts) * totalDemandStockoutProbs[s][d];
//		//		}
//		//	}
//		//}

//		double expStockouts = (1.0 - aggFillRate / totalDemandRate) * totalDemandRate;
//		serviceLevels.push_back(aggFillRate / totalDemandRate);
//		expectedPolicyHoldingCosts.push_back(expHoldingCost);
//		if (l == 9)
//			unavoidableCostPerPeriod = expHoldingCost;

//		//double expectedPenaltyCost = expExceededStockout * penaltyCost;
//		//double expectedTotalCost = expHoldingCost + expectedPenaltyCost;
//		std::cout << "Fr: " << serviceLevels[l];
//		for (int64_t stock : stockLevels) {
//			std::cout << "  " << stock;
//		}
//		std::cout << "  " << " holding costs:  " << expHoldingCost << " stockouts:  " << expStockouts * reviewHorizon << std::endl;

//		//std::cout << "  " << " holding costs:  " << expHoldingCost << " stockouts:  " << expStockouts * reviewHorizon << "  " << expExceededStockout;
//		//std::cout << " penalty cost:  " << expectedPenaltyCost << " total cost:  " << expectedTotalCost << std::endl;
//	}
//}

//void MDP::DetermineStockLevelsActionSet(std::vector<int64_t>& stockLevels, bool increase) const
//{
//	std::vector<double> changeFR(numberOfItems, 0.0);
//	std::vector<double> changeH(numberOfItems, 0.0);
//	std::vector<double> changeFRV(numberOfItems, 0.0);
//	for (size_t i = 0; i < numberOfItems; i++)
//	{
//		int64_t stock_level = increase ? stockLevels[i] : (stockLevels[i] - 1);
//		std::vector<double> statistics = CalculateItemStatistics(i, stock_level);
//		changeFR[i] = statistics[1];
//		changeH[i] = statistics[2];
//		changeFRV[i] = statistics[3];
//	}
//	double neg_limit = -std::numeric_limits<double>::infinity();
//	double pos_limit = std::numeric_limits<double>::infinity();
//	int64_t bestRatioSKU = 0;
//	for (size_t i = 0; i < numberOfItems; i++)
//	{
//		double ratio = changeFR[i] / (changeH[i] * holdingCosts[i]);
//		if (increase) {
//			if (ratio > neg_limit) {
//				bestRatioSKU = i;
//				neg_limit = ratio;
//			}

//		}
//		else {
//			if (ratio < pos_limit) {
//				bestRatioSKU = i;
//				pos_limit = ratio;
//			}
//		}
//	}
//	if (increase)
//		stockLevels[bestRatioSKU]++;
//	else
//		stockLevels[bestRatioSKU]--;
//}

//if (emergencyShipments) {
//	double rate = demandRates[i] * leadTimes[i];
//	double mult = 1.0;
//	for (int64_t j = 1; j <= stockLevels[i]; ++j) {
//		mult = (rate * mult) / (j + rate * mult);
//	}
//	aggFillRate += (1 - mult) * demandRates[i];
//}

//void MDP::DetermineStockLevels(std::vector<int64_t>& stockLevels, double serviceLevel) const
//{
//	double aggFillRate = 0.0;
//	std::vector<double> changeFR(numberOfItems, 0.0);
//	std::vector<double> changeH(numberOfItems, 0.0);
//	for (size_t i = 0; i < numberOfItems; i++)
//	{
//		if (emergencyShipments) {
//			double expFulfilled = 0.0;
//			double rate = demandRates[i] * leadTimes[i];
//			double mult = 1.0;
//			for (int64_t j = 1; j <= stockLevels[i]; ++j) {
//				mult = (rate * mult) / (j + rate * mult);
//			}
//			double fulFilled = (1 - mult) * demandRates[i];
//			expFulfilled += fulFilled;
//			mult = (rate * mult) / ((stockLevels[i] + 1) + rate * mult);
//			double newFulFilled = (1 - mult) * demandRates[i];
//			changeFR[i] = newFulFilled - fulFilled;
//			changeH[i] = 1.0;
//			aggFillRate += expFulfilled;
//		}
//		else {
//			std::vector<double> statistics = CalculateItemStatistics(i, stockLevels[i]);
//			aggFillRate += statistics[0];
//			changeFR[i] = statistics[1];
//			changeH[i] = statistics[2];
//		}
//	}
//	if (aggFillRate / totalDemandRate < serviceLevel)
//	{
//		double limit = -std::numeric_limits<double>::infinity();
//		int64_t bestRatioSKU{ 0 };
//		for (size_t i = 0; i < numberOfItems; i++)
//		{
//			double ratio = changeFR[i] / (changeH[i] * holdingCosts[i]);
//			if (ratio > limit)
//			{
//				bestRatioSKU = i;
//				limit = ratio;
//			}				
//		}
//		stockLevels[bestRatioSKU]++;
//		DetermineStockLevels(stockLevels, serviceLevel);
//	}
//}

//void MDP::DetermineStockLevelsContinuous(std::vector<int64_t>& stockLevels, double serviceLevel) const //book
//{
//	double aggFillRate{ 0.0 };
//	std::vector<double> probSums(numberOfItems, 0.0);
//	for (size_t i = 0; i < numberOfItems; i++)
//	{
//		for (size_t j = 0; j < stockLevels[i]; j++)
//		{
//			probSums[i] += demand_distributions_over_leadtime[i].ProbabilityAt(j);
//		}
//		aggFillRate += probSums[i] * demandRates[i] / totalDemandRate;
//	}
//	if (aggFillRate < serviceLevel)
//	{
//		double limit = -std::numeric_limits<double>::infinity();
//		int64_t bestRatioSKU{ 0 };
//		for (size_t i = 0; i < numberOfItems; i++)
//		{
//			double prob = demand_distributions_over_leadtime[i].ProbabilityAt(stockLevels[i]);
//			double ratio = (demandRates[i] / totalDemandRate) * prob / (holdingCosts[i] * (probSums[i] + prob));
//			if (ratio > limit)
//			{
//				bestRatioSKU = i;
//				limit = ratio;
//			}
//		}
//		stockLevels[bestRatioSKU]++;
//		DetermineStockLevels(stockLevels, serviceLevel);
//	}
//}
//std::vector<double> MDP::CalculateItemStatistics(int64_t item, int64_t stock_level) const
//{
//	std::vector<double> statistics(4, 0.0);
//	// expected fill rate, expected fill rate change, expected on hand inventory change, expected on hand inventory, variance of fill rate change
//	//double expFulfilledNewS = 0.0;
//	//double varDiff = 0.0;
//	for (int64_t j = stock_level; j >= 0; j--) {
//		double leadTimeDemandProb = demand_distributions_over_leadtime[item].ProbabilityAt(j);
//		double probSums = 0.0;
//		for (int64_t k = 0; k <= stock_level - j; k++) {
//			double prob = demand_distributions[item].ProbabilityAt(k);
//			probSums += prob;
//			statistics[0] += leadTimeDemandProb * prob * k;
//			//expFulfilledNewS += leadTimeDemandProb * prob * k;
//			statistics[3] += leadTimeDemandProb * prob * (stock_level - j - k);
//		}
//		double factor = leadTimeDemandProb * (1.0 - probSums);
//		statistics[0] += factor * (stock_level - j);
//		statistics[1] += factor;
//		statistics[2] += leadTimeDemandProb * probSums;
//		//expFulfilledNewS += factor * (stock_level - j + 1);
//		//varDiff += factor * (2.0 * stock_level - 2.0 * j + 1);
//	}
//	//statistics[4] = varDiff - expFulfilledNewS * expFulfilledNewS + statistics[0] * statistics[0];
//	return statistics;
//}
//
//void MDP::DetermineStockLevels(std::vector<int64_t>& stockLevels, double serviceLevel) const
//{
//	double aggFillRate = 0.0;
//	std::vector<double> changeFR(numberOfItems, 0.0);
//	//std::vector<double> changeFRV(numberOfItems, 0.0);
//	std::vector<double> changeH(numberOfItems, 0.0);
//	for (size_t i = 0; i < numberOfItems; i++)
//	{
//		std::vector<double> statistics = CalculateItemStatistics(i, stockLevels[i]);
//		aggFillRate += statistics[0];
//		changeFR[i] = statistics[1];
//		changeH[i] = statistics[2];
//		//changeFRV[i] = statistics[4];
//	}
//	if (aggFillRate / totalDemandRate < serviceLevel)
//	{
//		double limit = -std::numeric_limits<double>::infinity();
//		int64_t bestRatioSKU{ 0 };
//		for (size_t i = 0; i < numberOfItems; i++)
//		{
//			//double ratio = (changeFR[i] + meanVarRatio * std::max(0.0, changeFRV[i])) / (changeH[i] * holdingCosts[i]);
//			double ratio = changeFR[i] / (changeH[i] * holdingCosts[i]);
//			//std::cout << "ratio:  " << i << "  " << changeFR[i] << "   " << changeFRV[i] << "  " << (changeH[i] * holdingCosts[i]) << "  " << ratio << std::endl;
//			if (ratio > limit)
//			{
//				bestRatioSKU = i;
//				limit = ratio;
//			}
//		}
//		//std::cout << "Best: " << bestRatioSKU << "  fr:  " << aggFillRate / totalDemandRate << std::endl;
//		stockLevels[bestRatioSKU]++;
//		DetermineStockLevels(stockLevels, serviceLevel);
//	}
//}