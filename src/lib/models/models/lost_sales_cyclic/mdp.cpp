#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace lost_sales_cyclic /*keep this in line with id below and with namespace name in header*/
	{
		int64_t MDP::GetL(const State& state) const
		{
			return static_cast<int64_t>(state.limitingPeriod * 10);
		}

		int64_t MDP::GetH(const State& state) const
		{
			return static_cast<int64_t>(state.limitingPeriod * 5);
		}

		int64_t MDP::GetM(const State& state) const
		{
			return static_cast<int64_t>(state.limitingPeriod * 100);
		}

		int64_t MDP::GetReinitiateCounter(const State& state) const
		{
			return static_cast<int64_t>(state.limitingPeriod * 50);
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

		MDP::MDP(const VarGroup& config)
		{
			config.Get("evaluate", evaluate);
			config.Get("max_leadtime", max_leadtime);
			config.Get("max_demand", max_demand);
			config.Get("max_p", max_p);
			config.Get("max_num_cycles", max_num_cycles);
			h = 1.0;
			min_p = 2.0;
			min_leadtime = 0;
			min_demand = 2.0;
			returnRewards = false;

			if (evaluate) {
				config.Get("returnRewards", returnRewards);
				config.Get("censoredDemand", censoredDemand);
				config.Get("collectStatistics", collectStatistics);
				config.Get("p", p);
				config.Get("leadtime", leadtime);

				if (config.HasKey("demand_cycles"))
					config.Get("demand_cycles", demand_cycles);
				else
					demand_cycles = { 0 };

				if (demand_cycles.size() > 1) {
					config.Get("mean_cylic_demands", mean_demand);
					config.Get("std_cylic_demands", stdDemand);
				}
				else {
					double demand = 0.0;
					config.Get("mean_demand", demand);
					double stdev = 0.0;
					config.Get("stdDemand", stdev);
					mean_demand = { demand };
					stdDemand = { stdev };
				}
			}

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

		double MDP::ModifyStateWithAction(State& state, int64_t action) const
		{
			if (state.leadtime > 0)
				state.state_vector.push_back(action);
			else
				state.state_vector.front() += action;
			state.total_inv += action;
			state.cat = StateCategory::AwaitEvent();
			return 0.0;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
			if (state.order_initializationPhase > 0) {
				return action == state.MaxOrderSize;
			}
			else {
				return ((state.total_inv + action) <= state.MaxSystemInv && action <= state.MaxOrderSize) || action == 0;
			}
		}

		MDP::Event MDP::GetEvent(const State& state, RNG& rng) const {
			// Generate a uniform random number between 0 and 1
			double randomValue = rng.genUniform();
			double cumulativeProbability = 0.0;
			for (size_t i = 0; i < state.true_demand_probs[state.period].size(); i++) {
				cumulativeProbability += state.true_demand_probs[state.period][i];
				if (randomValue < cumulativeProbability) {
					return state.min_true_demand[state.period] + static_cast<int64_t>(i);
				}
			}
			return state.max_true_demand[state.period];
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

				if (state.leadtime > 0)
					state.state_vector.front() += onHand;
				else
					state.state_vector.push_back(onHand);

				cost = onHand * h;

				if (censoredDemand) {
					cost -= event * state.p;
				}
			}
			else
			{
				state.total_inv -= onHand;

				if (state.leadtime == 0)
					state.state_vector.push_back(0);

				if (censoredDemand) {
					uncensored = false;
					cost -= onHand * state.p;
					demand = onHand;
				} 
				else {
					cost = (event - onHand) * state.p;
				}
			}

			if (state.collectStatistics) {
				if (state.order_initializationPhase == 0)
					UpdateStatistics(state, uncensored, demand);
				else
					state.order_initializationPhase--;
			}
			else {
				if (state.demand_cycles.size() > 1) {
					state.period++;
					state.period = state.period % state.demand_cycles.size();
					state.MaxOrderSize = state.cycle_MaxOrderSize[state.period];
					state.MaxSystemInv = state.cycle_MaxSystemInv[state.period];
					state.fractiles = state.cycle_fractiles[state.period];
				}
			}

			if (returnRewards) {
				return -cost;
			}
			else {
				return cost;
			}
		}

		void MDP::UpdateStatistics(State& state, bool uncensored, int64_t newObs) const {
			int64_t current_cyclePeriod = state.demand_cycles[state.period];
			state.periodCount[current_cyclePeriod]++;
			state.period++;
			state.period = state.period % state.demand_cycles.size();

			int64_t oldSize = state.past_demands[current_cyclePeriod].size() - 1;
			if (newObs > oldSize) {
				for (int64_t i = oldSize + 1; i < newObs; i++) {
					state.past_demands[current_cyclePeriod].push_back(0);
					state.censor_indicator[current_cyclePeriod].push_back(0);
					state.cumulative_demands[current_cyclePeriod].push_back(1);
				}
				state.past_demands[current_cyclePeriod].push_back(1);
				state.cumulative_demands[current_cyclePeriod].push_back(0);

				if (uncensored) {
					state.censor_indicator[current_cyclePeriod].push_back(0);
				}
				else {
					state.censor_indicator[current_cyclePeriod].push_back(1);
				}
			}
			else {
				state.past_demands[current_cyclePeriod][newObs]++;
				oldSize = newObs - 1;

				if (!uncensored) {
					state.censor_indicator[current_cyclePeriod][newObs]++;
				}
			}
			for (int64_t i = 0; i < oldSize + 1; i++) {
				state.cumulative_demands[current_cyclePeriod][i]++;
			}

			int64_t demand_size = state.past_demands[current_cyclePeriod].size();
			std::vector<double> probs(demand_size, 0.0);
			// Iterative weight redistribution
			for (size_t i = 0; i < demand_size; i++) {
				probs[i] += static_cast<double>(state.past_demands[current_cyclePeriod][i]) / state.periodCount[current_cyclePeriod];
				if (state.censor_indicator[current_cyclePeriod][i] > 0 && i < demand_size - 1) { // Censored observation
					double weight_to_redistribute = static_cast<double>(state.censor_indicator[current_cyclePeriod][i]) / state.periodCount[current_cyclePeriod];
					probs[i] -= weight_to_redistribute;

					for (size_t j = i + 1; j < demand_size; ++j) {
						probs[j] += state.past_demands[current_cyclePeriod][j] * weight_to_redistribute / state.cumulative_demands[current_cyclePeriod][i];
					}
				}
			}

			state.cycle_probs[current_cyclePeriod] = probs;
			DynaPlex::DiscreteDist dist = DiscreteDist::GetCustomDist(probs, 0);
			state.mean_cycle_demand[current_cyclePeriod] = dist.Expectation();
			state.std_cycle_demand[current_cyclePeriod] = dist.StandardDeviation();

			auto DummyDemOverLeadtime = DiscreteDist::GetZeroDist();
			for (size_t i = 0; i <= state.leadtime; i++)
			{
				int64_t cyclePeriod_over_leadtime = state.demand_cycles[(state.period + i) % state.demand_cycles.size()];
				DynaPlex::DiscreteDist dist_over_leadtime = DiscreteDist::GetCustomDist(state.cycle_probs[cyclePeriod_over_leadtime], 0);
				DummyDemOverLeadtime = DummyDemOverLeadtime.Add(dist_over_leadtime);

				if (i == state.leadtime) {
					state.MaxOrderSize = std::min(dist_over_leadtime.Fractile(state.p / (state.p + h)), MaxOrderSize);
					for (size_t i = 0; i < fractiles.size(); i++) {
						state.fractiles[i] = dist_over_leadtime.Fractile(fractiles[i]);
					}
				}
			}
			state.MaxSystemInv = std::min(DummyDemOverLeadtime.Fractile(state.p / (state.p + h)), MaxSystemInv);		
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			features.Add(state.state_vector);
			for (int64_t i = 0; i < max_leadtime - std::max((int64_t)1, state.leadtime); i++) {
				features.Add(0);
			}
			features.Add(state.leadtime);
			features.Add(state.p);
			features.Add(state.fractiles);
			for (int64_t i = 0; i < max_leadtime; i++) {
				int64_t cyclePeriod = state.demand_cycles[(state.period + i) % state.demand_cycles.size()];
				features.Add(state.mean_cycle_demand[cyclePeriod]);
				features.Add(state.std_cycle_demand[cyclePeriod]);
			}
		}

		MDP::State MDP::GetInitialState(RNG& rng) const
		{
			State state{};

			state.period = 0;
			state.order_initializationPhase = 0;
			state.collectStatistics = false;
			std::vector<double> mean_true_demand;
			std::vector<double> stdev_true_demand;

			if (evaluate) {
				state.demand_cycles = demand_cycles;
				mean_true_demand = mean_demand;
				stdev_true_demand = stdDemand;
				state.p = p;
				state.leadtime = leadtime;

				if (collectStatistics) {
					state.mean_cycle_demand.reserve(state.demand_cycles.size());
					state.std_cycle_demand.reserve(state.demand_cycles.size());

					if (censoredDemand) {
						state.periodCount.reserve(state.demand_cycles.size());
						for (int64_t i = 0; i < state.demand_cycles.size(); i++) {
							state.mean_cycle_demand.push_back(0.0);
							state.std_cycle_demand.push_back(0.0);
							state.periodCount.push_back(0);
							state.past_demands.push_back({});
							state.cumulative_demands.push_back({});
							state.censor_indicator.push_back({});
							state.cycle_probs.push_back({});
							state.collectStatistics = true;
							state.order_initializationPhase = std::max(state.leadtime, (int64_t)state.demand_cycles.size());
						}
					}
					else {
						for (int64_t i = 0; i < state.demand_cycles.size(); i++) {
							state.mean_cycle_demand.push_back(mean_true_demand[i]);
							state.std_cycle_demand.push_back(stdev_true_demand[i]);
						}
					}
				}
			}
			else {
				double randomValue_p = rng.genUniform();
				state.p = randomValue_p * (max_p - min_p) + min_p;
				double randomValue_leadtime = rng.genUniform();
				state.leadtime = static_cast<int64_t>(std::floor(randomValue_leadtime * (max_leadtime - min_leadtime + 1))) + min_leadtime;
				double randomValue_cycles = rng.genUniform();
				int64_t num_cycles = static_cast<int64_t>(std::floor(randomValue_leadtime * max_num_cycles)) + 1;
				mean_true_demand.reserve(num_cycles);
				stdev_true_demand.reserve(num_cycles);
				state.demand_cycles.reserve(num_cycles);
				for (int64_t i = 0; i < num_cycles; i++) {
					state.demand_cycles.push_back(i);
					double mean = rng.genUniform() * (max_demand - min_demand) + min_demand;
					mean_true_demand.push_back(mean);
					int64_t n = static_cast<int64_t>(std::round(mean / 0.2));
					double prob = mean / n;
					double binom_stdev = std::sqrt(n * prob * (1.0 - prob));
					stdev_true_demand.push_back(rng.genUniform() * (mean * 2.0 - binom_stdev) + binom_stdev);
				}
				state.mean_cycle_demand = mean_true_demand;
				state.std_cycle_demand = stdev_true_demand;
				state.limitingPeriod = std::max(state.leadtime, (int64_t)state.demand_cycles.size());
			}

			auto queue = Queue<int64_t>{};
			queue.reserve(state.leadtime + 1);
			queue.push_back(0);
			for (int64_t i = 0; i < state.leadtime - 1; i++)
			{
				queue.push_back(0);
			}
			state.cat = StateCategory::AwaitAction();
			state.state_vector = queue;
			state.total_inv = queue.sum();

			state.true_demand_probs.reserve(state.demand_cycles.size());
			state.min_true_demand.reserve(state.demand_cycles.size());
			state.max_true_demand.reserve(state.demand_cycles.size());
			for (int64_t i = 0; i < state.demand_cycles.size(); i++) {
				DynaPlex::DiscreteDist state_demand_dist = DiscreteDist::GetAdanEenigeResingDist(mean_true_demand[i], stdev_true_demand[i]);
				state.min_true_demand.push_back(state_demand_dist.Min());
				state.max_true_demand.push_back(state_demand_dist.Max());
				std::vector<double> probs;
				probs.reserve(state_demand_dist.DistinctValueCount());
				for (const auto& [qty, prob] : state_demand_dist) {
					probs.push_back(prob);
				}
				state.true_demand_probs.push_back(probs);
			}

			if (!evaluate || (evaluate && !state.collectStatistics)) {
				state.cycle_fractiles.reserve(state.demand_cycles.size());
				state.cycle_MaxOrderSize.reserve(state.demand_cycles.size());
				state.cycle_MaxSystemInv.reserve(state.demand_cycles.size());
				for (int64_t i = 0; i < state.demand_cycles.size(); i++) {
					auto DummyDemOverLeadtime = DiscreteDist::GetZeroDist();
					for (int64_t j = 0; j <= state.leadtime; j++)
					{
						int64_t cyclePeriod_over_leadtime = (i + j) % state.demand_cycles.size();
						DynaPlex::DiscreteDist dist_over_leadtime = DiscreteDist::GetCustomDist(state.true_demand_probs[cyclePeriod_over_leadtime], state.min_true_demand[cyclePeriod_over_leadtime]);
						DummyDemOverLeadtime = DummyDemOverLeadtime.Add(dist_over_leadtime);

						if (j == state.leadtime) {
							state.cycle_MaxOrderSize.push_back(std::min(MaxOrderSize, dist_over_leadtime.Fractile(state.p / (state.p + h))));
							std::vector<int64_t> _fractiles(fractiles.size(), 0);
							for (size_t i = 0; i < fractiles.size(); i++) {
								_fractiles[i] = dist_over_leadtime.Fractile(fractiles[i]);
							}
							state.cycle_fractiles.push_back(_fractiles);
						}
					}
					state.cycle_MaxSystemInv.push_back(std::min(MaxSystemInv, DummyDemOverLeadtime.Fractile(state.p / (state.p + h))));
				}
				state.MaxOrderSize = state.cycle_MaxOrderSize[state.period];
				state.MaxSystemInv = state.cycle_MaxSystemInv[state.period];
				state.fractiles = state.cycle_fractiles[state.period];
			}
			else {
				auto DemOverLeadtime = DiscreteDist::GetZeroDist();
				DynaPlex::DiscreteDist edge_dist = DiscreteDist::GetAdanEenigeResingDist(max_demand, max_demand * 2);
				for (size_t i = 0; i <= state.leadtime; i++)
				{
					DemOverLeadtime = DemOverLeadtime.Add(edge_dist);
				}
				state.MaxOrderSize = std::min(MaxOrderSize, edge_dist.Fractile(state.p / (state.p + h)));
				state.MaxSystemInv = std::min(MaxSystemInv, DemOverLeadtime.Fractile(state.p / (state.p + h)));
				std::vector<int64_t> _fractiles(fractiles.size(), 0);
				state.fractiles = _fractiles;
				for (size_t i = 0; i < fractiles.size(); i++) {
					state.fractiles[i] = edge_dist.Fractile(fractiles[i]);
				}
			}

			return state;
		}

		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("state_vector", state.state_vector);
			vars.Get("total_inv", state.total_inv);
			vars.Get("demand_cycles", state.demand_cycles);
			vars.Get("mean_cycle_demand", state.mean_cycle_demand);
			vars.Get("std_cycle_demand", state.std_cycle_demand);
			vars.Get("fractiles", state.fractiles);
			vars.Get("collectStatistics", state.collectStatistics);
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
			vars.Add("demand_cycles", demand_cycles);
			vars.Add("mean_cycle_demand", mean_cycle_demand);
			vars.Add("std_cycle_demand", std_cycle_demand);
			vars.Add("fractiles", fractiles);
			vars.Add("collectStatistics", collectStatistics);
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
				/*=id though which the MDP will be retrievable*/ "lost_sales_cyclic",
				/*description*/ "Lost sales problem with cyclic censored demand.)",
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

