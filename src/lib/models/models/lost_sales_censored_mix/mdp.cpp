#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace lost_sales_censored_mix /*keep this in line with id below and with namespace name in header*/
	{
		int64_t MDP::GetL(const State& state) const
		{
			return static_cast<int64_t>(state.leadtime * 10);		
		}

		int64_t MDP::GetH(const State& state) const
		{
			return static_cast<int64_t>(state.leadtime * 5);
		}

		int64_t MDP::GetM(const State& state) const
		{
			return static_cast<int64_t>(state.leadtime * 100);
		}

		int64_t MDP::GetReinitiateCounter(const State& state) const
		{
			return static_cast<int64_t>(state.leadtime * 50);
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
			if (state.collectStatistics && state.periodCount <= state.leadtime) {
				return action == state.MaxOrderSize;
			}
			else {
				return ((state.total_inv + action) <= state.MaxSystemInv && action <= state.MaxOrderSize) || action == 0;
			}
		}

		MDP::MDP(const VarGroup& config)
		{
			config.Get("max_p", max_p);
			config.Get("max_leadtime", max_leadtime);
			config.Get("collectStatistics", collectStatistics);
			config.Get("evaluate", evaluate);
			config.Get("returnRewards", returnRewards);
			config.Get("censoredDemand", censoredDemand);
			
			h = 1.0;
			min_p = 2.0;
			min_leadtime = 2;

			if (config.HasKey("leadtime"))
				config.Get("leadtime", leadtime);
			else
				leadtime = min_leadtime;

			if (config.HasKey("p"))
				config.Get("p", p);
			else
				p = min_p;

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

			DynaPlex::DiscreteDist dist = DiscreteDist::GetAdanEenigeResingDist(max_demand, max_demand * 2);
			//Initiate members that are computed from the parameters:
			auto DemOverLeadtime = DiscreteDist::GetZeroDist();
			for (size_t i = 0; i <= max_leadtime; i++)
			{
				DemOverLeadtime = DemOverLeadtime.Add(dist);
			}
			MaxOrderSize = dist.Fractile(max_p / (max_p + h));
			MaxSystemInv = DemOverLeadtime.Fractile(max_p / (max_p + h));
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
			int64_t demand = event;
			bool uncensored = true;
			double cost{ 0.0 };

			if (onHand >= event)
			{
				onHand -= event;
				state.total_inv -= event;
				state.state_vector.front() += onHand;
				cost = onHand * h;

				if (censoredDemand) {
					cost -= event * state.p;
				}
			}
			else
			{
				state.total_inv -= onHand;
				uncensored = false;
				demand = onHand;

				if (!censoredDemand) {
					cost = (event - onHand) * state.p;
				} 
				else {
					cost -= onHand * state.p;
				}
			}

			if (state.collectStatistics) {
				state.periodCount++;
				UpdateStatistics(state, uncensored, demand);
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

			auto DummyDemOverLeadtime = DiscreteDist::GetZeroDist();
			for (size_t i = 0; i <= state.leadtime; i++)
			{
				DummyDemOverLeadtime = DummyDemOverLeadtime.Add(dist);
			}
			int64_t DummyMaxOrderSize = dist.Fractile(state.p / (state.p + h));
			int64_t DummyMaxSystemInv = DummyDemOverLeadtime.Fractile(state.p / (state.p + h));
			state.MaxOrderSize = std::min(DummyMaxOrderSize, MaxOrderSize);
			state.MaxSystemInv = std::min(DummyMaxSystemInv, MaxSystemInv);
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			features.Add(state.state_vector);
			for (int64_t i = 0; i < max_leadtime - state.leadtime; i++) {
				features.Add(0);
			}
			features.Add(state.leadtime);
			features.Add(state.p);
			features.Add(state.fractiles);
			features.Add(state.meanUncensoredDemand);
			features.Add(state.stdUncensoredDemand);
		}

		MDP::State MDP::GetInitialState(RNG& rng) const
		{
			State state{};

			state.periodCount = 0;
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
				state.p = p;
				state.leadtime = leadtime;
				if (censoredDemand) {
					state.meanUncensoredDemand = 0.0;
					state.stdUncensoredDemand = 0.0;
					state.collectStatistics = collectStatistics;
				}
				else {
					state.meanUncensoredDemand = mean_true_demand;
					state.stdUncensoredDemand = stdev_true_demand;
					state.collectStatistics = false;
				}
			}
			else {
				double randomValue_p = rng.genUniform();
				state.p = randomValue_p * (max_p - min_p) + min_p;
				double randomValue_leadtime = rng.genUniform();
				state.leadtime = static_cast<int64_t>(std::floor(randomValue_leadtime * (max_leadtime - min_leadtime + 1))) + min_leadtime;
				double randomValue_mean = rng.genUniform();
				mean_true_demand = randomValue_mean * (max_demand - min_demand) + min_demand;
				double randomValue_stdev = rng.genUniform();
				double p_dummy = 0.2;
				int64_t n = static_cast<int64_t>(std::round(mean_true_demand / p_dummy));
				double prob = mean_true_demand / n;
				double var = n * prob * (1 - prob);
				double binom_stdev = std::sqrt(var);
				stdev_true_demand = randomValue_stdev * (mean_true_demand * 2 - binom_stdev) + binom_stdev;
				state.collectStatistics = false;	
				state.meanUncensoredDemand = mean_true_demand;
				state.stdUncensoredDemand = stdev_true_demand;
			}

			auto queue = Queue<int64_t>{};
			queue.reserve(state.leadtime + 1);
			for (size_t i = 0; i < state.leadtime; i++)
			{
				queue.push_back(0);
			}
			state.cat = StateCategory::AwaitAction();
			state.state_vector = queue;
			state.total_inv = queue.sum();

			DynaPlex::DiscreteDist state_demand_dist = DiscreteDist::GetAdanEenigeResingDist(mean_true_demand, stdev_true_demand);
			state.min_true_demand = state_demand_dist.Min();
			state.max_true_demand = state_demand_dist.Max();
			state.true_demand_probs.reserve(state_demand_dist.DistinctValueCount());
			for (const auto& [qty, prob] : state_demand_dist) {
				state.true_demand_probs.push_back(prob);
			}
			for (size_t i = 0; i < fractiles.size(); i++) {
				state.fractiles[i] = state_demand_dist.Fractile(fractiles[i]);
			}

			auto DemOverLeadtime = DiscreteDist::GetZeroDist();
			if (state.collectStatistics) {
				DynaPlex::DiscreteDist dist = DiscreteDist::GetAdanEenigeResingDist(max_demand, max_demand * 2);
				for (size_t i = 0; i <= state.leadtime; i++)
				{
					DemOverLeadtime = DemOverLeadtime.Add(dist);
				}
				state.MaxOrderSize = dist.Fractile(state.p / (state.p + h));
				state.MaxSystemInv = DemOverLeadtime.Fractile(state.p / (state.p + h));
			}
			else {
				for (size_t i = 0; i <= state.leadtime; i++)
				{
					DemOverLeadtime = DemOverLeadtime.Add(state_demand_dist);
				}
				state.MaxOrderSize = state_demand_dist.Fractile(state.p / (state.p + h));
				state.MaxSystemInv = DemOverLeadtime.Fractile(state.p / (state.p + h));
			}
			state.MaxOrderSize = std::min(MaxOrderSize, state.MaxOrderSize);
			state.MaxSystemInv = std::min(MaxSystemInv, state.MaxSystemInv);

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
			vars.Get("meanUncensoredDemand", state.meanUncensoredDemand);
			vars.Get("stdUncensoredDemand", state.stdUncensoredDemand);
			vars.Get("collectStatistics", state.collectStatistics);
			vars.Get("MaxSystemInv", state.MaxSystemInv);
			vars.Get("MaxOrderSize", state.MaxOrderSize);
			vars.Get("p", state.p);
			vars.Get("leadtime", state.leadtime);
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
			vars.Add("meanUncensoredDemand", meanUncensoredDemand);
			vars.Add("stdUncensoredDemand", stdUncensoredDemand);
			vars.Add("collectStatistics", collectStatistics);
			vars.Add("MaxSystemInv", MaxSystemInv);
			vars.Add("MaxOrderSize", MaxOrderSize);
			vars.Add("p", p);
			vars.Add("leadtime", leadtime);
			return vars;
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "lost_sales_censored_mix",
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
	}
}

