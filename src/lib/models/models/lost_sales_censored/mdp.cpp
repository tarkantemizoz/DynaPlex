#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace lost_sales_censored /*keep this in line with id below and with namespace name in header*/
	{
		int64_t MDP::GetReinitiateCounter(const State& state) const
		{
			//if (state.periodCount < 200) {
				return 500;
			//}
			//else {
			//	return 0;
			//}
		}

		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;		
			vars.Add("valid_actions", MaxOrderSize + 1);
			vars.Add("discount_factor", discount_factor);

			VarGroup diagnostics{};			
			diagnostics.Add("MaxOrderSize", MaxOrderSize);
			diagnostics.Add("MaxSystemInv", MaxSystemInv);
			vars.Add("diagnostics", diagnostics);
			
			return vars;
		}

		double MDP::ModifyStateWithAction(State& state, int64_t action) const
		{
			state.state_vector.push_back(action);
			state.total_inv += action;
			state.cat = StateCategory::AwaitEvent();
			return 0.0;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
			if (state.initialperiodCount <= leadtime) {
				return action == state.MaxOrderSize;
			}
			else {
				return ((state.total_inv + action) <= state.MaxSystemInv && action <= state.MaxOrderSize && (state.total_inv + action) >= state.MinSystemInv) || action == 0;
			}
		}

		MDP::MDP(const VarGroup& config)
		{
			config.Get("p", p);
			config.Get("leadtime", leadtime);
			config.Get("collectStatistics", collectStatistics);
			config.Get("evaluate", evaluate);
			config.Get("returnRewards", returnRewards);
			
			h = 1.0;

			if (config.HasKey("updateCycle"))
				config.Get("updateCycle", updateCycle);
			else
				updateCycle = 5;

			if (config.HasKey("statTreshold"))
				config.Get("statTreshold", statTreshold);
			else
				statTreshold = 30;

			if (config.HasKey("max_demand"))
				config.Get("max_demand", max_demand);
			else
				max_demand = 10.0;

			if (config.HasKey("min_demand"))
				config.Get("min_demand", min_demand);
			else
				min_demand = 2.0;

			if (config.HasKey("mean_demand"))
				config.Get("mean_demand", mean_demand);
			else
				mean_demand = max_demand;

			if (config.HasKey("stdDemand"))
				config.Get("stdDemand", stdDemand);
			else
				stdDemand = std::sqrt(mean_demand);

			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;
			
			fractiles = { 0.09, 0.19, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.89, 0.99 };
			MinSystemInv = 0;

			double demand = min_demand;
			while (demand <= max_demand) {
				demand_vector.push_back(demand);
				double max_std = demand * 2;
				double p_dummy = 0.2;
				int64_t n = static_cast<int64_t>(std::round(demand / p_dummy));
				double prob = demand / n;
				double var = n * prob * (1 - prob);
				double stdev = ceil(std::sqrt(var));
				std::vector<int64_t> max_orders;
				std::vector<int64_t> max_invs;
				std::vector<double> stds;
				while (stdev <= max_std) {
					stds.push_back(stdev);
					DynaPlex::DiscreteDist dummy_demand_dist = DiscreteDist::GetAdanEenigeResingDist(demand, stdev);
					auto DummyDemOverLeadtime = DiscreteDist::GetZeroDist();
					for (size_t i = 0; i <= leadtime; i++)
					{
						DummyDemOverLeadtime = DummyDemOverLeadtime.Add(dummy_demand_dist);
					}
					int64_t DummyMaxOrderSize = dummy_demand_dist.Fractile(p / (p + h));
					int64_t DummyMaxSystemInv = DummyDemOverLeadtime.Fractile(p / (p + h));
					max_orders.push_back(DummyMaxOrderSize);
					max_invs.push_back(DummyMaxSystemInv);
					stdev += 1.0;
				}
				std_vector.push_back(stds);
				ordervector.push_back(max_orders);
				invvector.push_back(max_invs);
				demand += 1.0;
			}

			MaxOrderSize = ordervector[demand_vector.size() - 1][std_vector[demand_vector.size() - 1].size() - 1];
			MaxSystemInv = invvector[demand_vector.size() - 1][std_vector[demand_vector.size() - 1].size() - 1];
		}

		MDP::Event MDP::GetEvent(const State& state, RNG& rng) const {
			// Generate a uniform random number between 0 and 1
			double randomValue = rng.genUniform();
			double cumulativeProbability = 0.0;
			for (size_t i = 0; i < state.true_demand_probs.size(); i++) {
				cumulativeProbability += state.true_demand_probs[i];
				if (randomValue < cumulativeProbability) {
					return state.min_true_demand + static_cast<int64_t>(i);
				}
			}
			return state.max_true_demand;
		}

		double MDP::ModifyStateWithEvent(State& state,const MDP::Event& event) const
		{
			state.cat = StateCategory::AwaitAction();
			auto onHand = state.state_vector.pop_front();
			double cost{ 0.0 };
			if (state.collectStatistics) {
				state.initialperiodCount++;
			}

			if (onHand >= event)
			{
				onHand -= event;
				state.total_inv -= event;
				state.state_vector.front() += onHand;
				cost = onHand * h;
				if (state.initialperiodCount > leadtime) {
					state.periodCount++;
					state.uncensoredperiodCount++;
					UpdateStatistics(state, true, event);
				}
				cost -= event * p;
			}
			else
			{
				state.total_inv -= onHand;
				if (state.initialperiodCount > leadtime) {
					state.periodCount++;
					UpdateStatistics(state, false, onHand);
				}
				cost -= onHand * p;
			}

			if (returnRewards) {
				return -cost;
			}
			else {
				return cost;
			}
		}

		void MDP::UpdateStatistics(State& state, bool uncensored, int64_t newObs) const {
			int64_t oldSize = state.past_demands.size() - 1;
			if (newObs > oldSize) {
				for (int64_t i = oldSize + 1; i < newObs; i++) {
					state.past_demands.push_back(0);
					state.censor_indicator.push_back(0);
					state.cumulative_demands.push_back(1);
				}
				state.past_demands.push_back(1);
				state.cumulative_demands.push_back(0);

				if (uncensored) {
					state.censor_indicator.push_back(0);
				}
				else {
					state.censor_indicator.push_back(1);
				}
			}
			else {
				state.past_demands[newObs]++;
				oldSize = newObs - 1;

				if (!uncensored) {
					state.censor_indicator[newObs]++;
				}
			}
			for (int64_t i = 0; i < oldSize + 1; i++) {
				state.cumulative_demands[i]++;
			}

			if (state.collectStatistics || state.uncensoredperiodCount == state.statTreshold) {
				int64_t demand_size = state.past_demands.size();
				std::vector<double> probs(demand_size, 0.0);
				// Iterative weight redistribution
				for (size_t i = 0; i < demand_size; i++) {
					probs[i] += static_cast<double>(state.past_demands[i]) / state.periodCount;
					if (state.censor_indicator[i] > 0 && i < demand_size - 1) { // Censored observation
						double weight_to_redistribute = static_cast<double>(state.censor_indicator[i]) / state.periodCount;
						probs[i] -= weight_to_redistribute;

						for (size_t j = i + 1; j < demand_size; ++j) {
							probs[j] += state.past_demands[j] * weight_to_redistribute / state.cumulative_demands[i];
						}
					}
				}

				DynaPlex::DiscreteDist dist = DiscreteDist::GetCustomDist(probs, 0);
				state.meanUncensoredDemand = dist.Expectation();
				state.stdUncensoredDemand = std::sqrt(dist.Variance());
				for (size_t i = 0; i < fractiles.size(); i++) {
					state.fractiles[i] = dist.Fractile(fractiles[i]);
				}

				//if (state.initialperiodCount > updateCycle) {
				int64_t demand_loc = findGreaterIndex(demand_vector, state.meanUncensoredDemand);
				int64_t std_loc = findGreaterIndex(std_vector[demand_loc], state.stdUncensoredDemand);
				state.MaxOrderSize = std::min(ordervector[demand_loc][std_loc], MaxOrderSize);
				state.MaxSystemInv = std::min(invvector[demand_loc][std_loc], MaxSystemInv);
				//}

				if (state.uncensoredperiodCount == state.statTreshold) {
					state.collectStatistics = false;
					state.uncensoredperiodCount = 0;
				}

				if (state.collectStatistics) {
					if (state.periodCount > 1) {
						double newMean = (state.meanDemand * (state.periodCount - 1) + newObs) / state.periodCount;
						double newVariance = ((state.stdDemand * state.stdDemand * (state.periodCount - 1)) + (newObs - state.meanDemand) * (newObs - newMean)) / state.periodCount;
						state.meanDemand = newMean;
						state.stdDemand = std::sqrt(newVariance);
					}
					else {
						state.meanDemand = newObs;
					}
				}
			}
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			features.Add(state.state_vector);
			if (state.collectStatistics) {
				features.Add(state.fractiles);
				features.Add(state.meanUncensoredDemand);
				features.Add(state.stdUncensoredDemand);
				//features.Add(state.uncensoredperiodCount);
				//features.Add(state.meanDemand);
				//features.Add(state.stdDemand);
				//features.Add(state.periodCount);
			}
			else {
				features.Add(state.fractiles);
				features.Add(state.meanUncensoredDemand);
				features.Add(state.stdUncensoredDemand);
				//features.Add(0);
				//features.Add(0);
				//features.Add(0);
				//features.Add(0);
			}
		}

		MDP::State MDP::GetInitialState(RNG& rng) const
		{
			State state{};

			state.periodCount = 0;
			state.uncensoredperiodCount = 0;
			state.initialperiodCount = 0;
			state.meanDemand = 0.0;
			state.stdDemand = 0.0;
			state.meanUncensoredDemand = 0.0;
			state.stdUncensoredDemand = 0.0;
			state.collectStatistics = collectStatistics;
			state.statTreshold = statTreshold;
			state.updateCycle = updateCycle;

			auto queue = Queue<int64_t>{};
			queue.reserve(leadtime + 1);
			for (size_t i = 0; i < leadtime; i++)
			{
				queue.push_back(0);
			}
			state.cat = StateCategory::AwaitAction();
			state.state_vector = queue;
			state.total_inv = queue.sum();

			state.past_demands = {};
			state.cumulative_demands = {};
			state.censor_indicator = {};
			std::vector<int64_t> _fractiles(fractiles.size(), 0);
			state.fractiles = _fractiles;

			double mean_true_demand{ 0.0 };
			double stdev_true_demand{ 0.0 };

			if (evaluate) {
				mean_true_demand = mean_demand;
				stdev_true_demand = stdDemand;
			}
			else {
				double randomValue_mean = rng.genUniform();
				mean_true_demand = randomValue_mean * (max_demand - min_demand) + min_demand;
				double randomValue_stdev = rng.genUniform();
				double p_dummy = 0.2;
				int64_t n = static_cast<int64_t>(std::round(mean_true_demand / p_dummy));
				double prob = mean_true_demand / n;
				double var = n * prob * (1 - prob);
				double binom_stdev = std::sqrt(var);
				stdev_true_demand = randomValue_stdev * (mean_true_demand * 2 - binom_stdev) + binom_stdev;
			}

			DynaPlex::DiscreteDist state_demand_dist = DiscreteDist::GetAdanEenigeResingDist(mean_true_demand, stdev_true_demand);
			state.min_true_demand = state_demand_dist.Min();
			state.max_true_demand = state_demand_dist.Max();
			state.true_demand_probs.reserve(state_demand_dist.DistinctValueCount());
			for (const auto& [qty, prob] : state_demand_dist) {
				state.true_demand_probs.push_back(prob);
			}

			state.MaxOrderSize = MaxOrderSize;
			state.MaxSystemInv = MaxSystemInv;
			state.MinSystemInv = MinSystemInv;

			return state;
		}

		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("state_vector", state.state_vector);
			vars.Get("total_inv", state.total_inv);
			vars.Get("min_true_demand", state.min_true_demand);
			vars.Get("max_true_demand", state.max_true_demand);
			vars.Get("true_demand_probs", state.true_demand_probs);
			vars.Get("past_demands", state.past_demands);
			vars.Get("cumulative_demands", state.cumulative_demands);
			vars.Get("censor_indicator", state.censor_indicator);
			vars.Get("fractiles", state.fractiles);
			vars.Get("periodCount", state.periodCount);
			vars.Get("uncensoredperiodCount", state.uncensoredperiodCount);
			vars.Get("initialperiodCount", state.initialperiodCount);
			vars.Get("meanDemand", state.meanDemand);
			vars.Get("stdDemand", state.stdDemand);
			vars.Get("meanUncensoredDemand", state.meanUncensoredDemand);
			vars.Get("stdUncensoredDemand", state.stdUncensoredDemand);
			vars.Get("collectStatistics", state.collectStatistics);
			vars.Get("MaxSystemInv", state.MaxSystemInv);
			vars.Get("MinSystemInv", state.MinSystemInv);
			vars.Get("MaxOrderSize", state.MaxOrderSize);
			vars.Get("statTreshold", state.statTreshold);
			vars.Get("updateCycle", state.updateCycle);
			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("state_vector", state_vector);
			vars.Add("total_inv", total_inv);
			vars.Add("min_true_demand", min_true_demand);
			vars.Add("max_true_demand", max_true_demand);
			vars.Add("true_demand_probs", true_demand_probs);
			vars.Add("past_demands", past_demands);
			vars.Add("cumulative_demands", cumulative_demands);
			vars.Add("censor_indicator", censor_indicator);
			vars.Add("fractiles", fractiles);
			vars.Add("periodCount", periodCount);
			vars.Add("uncensoredperiodCount", uncensoredperiodCount);
			vars.Add("initialperiodCount", initialperiodCount);
			vars.Add("meanDemand", meanDemand);
			vars.Add("stdDemand", stdDemand);
			vars.Add("meanUncensoredDemand", meanUncensoredDemand);
			vars.Add("stdUncensoredDemand", stdUncensoredDemand);
			vars.Add("collectStatistics", collectStatistics);
			vars.Add("MaxSystemInv", MaxSystemInv);
			vars.Add("MinSystemInv", MinSystemInv);
			vars.Add("MaxOrderSize", MaxOrderSize);
			vars.Add("statTreshold", statTreshold);
			vars.Add("updateCycle", updateCycle);
			return vars;
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "lost_sales_censored",
				/*description*/ "Lost sales problem with censored demand.)",
				/*reference to passed registry*/registry); 
		}

		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{
			registry.Register<BaseStockPolicy>("base_stock",
				"Oracle base-stock policy with parameter S.");
			registry.Register<CappedBaseStockPolicy>("capped_base_stock",
				"Oracle capped base-stock policy with parameters S and r.");
			registry.Register<GreedyCappedBaseStockPolicy>("greedy_capped_base_stock",
				"Capped base-stock policy with suboptimal S and r.");
			registry.Register<ConstantOrderPolicy>("constant_order",
				"Constant order policy with parameter co_level.");
		}

		int64_t MDP::findGreaterIndex(const std::vector<double>& v, double x) const {
			auto it = std::upper_bound(v.begin(), v.end(), x); // Find first element greater than x
			if (it == v.end()) {
				// x is greater than all elements, return index of last element
				return v.size() - 1;
			}

			return it - v.begin(); // Return index of the found element
		}
	}
}

