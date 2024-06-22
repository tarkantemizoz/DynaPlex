#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace lost_sales_all /*keep this in line with id below and with namespace name in header*/
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
				config.Get("censoredLeadtime", censoredLeadtime);
				config.Get("collectStatistics", collectStatistics);
				config.Get("p", p);

				if (config.HasKey("demand_cycles"))
					config.Get("demand_cycles", demand_cycles);
				else
					demand_cycles = { 0 };

				if (demand_cycles.size() > 1) {
					config.Get("mean_cylic_demands", mean_demand);
					config.Get("std_cylic_demands", stdDemand);
					if (mean_demand.size() != demand_cycles.size() || stdDemand.size() != demand_cycles.size())
						throw DynaPlex::Error("Size of mean/std demand should be equal to demand cycle size.");
				}
				else {
					double demand = 0.0;
					config.Get("mean_demand", demand);
					double stdev = 0.0;
					config.Get("stdDemand", stdev);
					mean_demand = { demand };
					stdDemand = { stdev };
				}
				
				std::vector<double> probs(max_leadtime + 1, 0.0);
				leadtime_probs = probs;
				int64_t leadtime;
				if (config.HasKey("leadtime")) {
					config.Get("leadtime", leadtime);
					if (leadtime > max_leadtime || leadtime < min_leadtime)
						throw DynaPlex::Error("Leadtime should be between max_leadtime and min_leadtime.");
					else
						leadtime_probs[leadtime] = 1.0;
				}
				else if (config.HasKey("leadtime_probs")) {
					config.Get("leadtime_probs", leadtime_probs);
					if (leadtime_probs.size() != max_leadtime + 1)
						throw DynaPlex::Error("Size of leadtime probability vector should be max_leadtime + 1.");
					double total_prob = 0.0;
					for (int64_t i = 0; i < leadtime_probs.size(); i++) {
						total_prob += leadtime_probs[i];
					}
					if (std::abs(total_prob - 1.0) >= 1e-8)
						throw DynaPlex::Error("Evaluate: Total lead time probabilities should sum up to 1.0.");
				}
				else {
					throw DynaPlex::Error("Provide a leadtime value or leadtime distribution.");
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
			state.state_vector.push_back(action);
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
			double randomValue = rng.genUniform();
			double cumulativeProbability = 0.0;
			int64_t demand = state.max_true_demand[state.period];
			for (size_t i = 0; i < state.true_demand_probs[state.period].size(); i++) {
				cumulativeProbability += state.true_demand_probs[state.period][i];
				if (randomValue < cumulativeProbability) {
					demand = state.min_true_demand[state.period] + static_cast<int64_t>(i);
					break;
				}
			}

			std::vector<double> arrival_prob{};
			if (state.stochasticLeadtimes) {
				for (int64_t j = state.min_leadtime; j <= state.max_leadtime; j++) {
					if (j == 0) {
						for (int64_t i = 0; i < state.state_vector.back(); i++) {
							arrival_prob.push_back(rng.genUniform());
						}
						for (int64_t i = state.state_vector.back(); i < state.MaxOrderSize_Limit; i++) {
							rng.genUniform();
						}
					}
					else {
						int64_t pipeline_inv = state.state_vector.at(max_leadtime + 1 - j);
						for (int64_t i = 0; i < pipeline_inv; i++) {
							arrival_prob.push_back(rng.genUniform());
						}
						for (int64_t i = pipeline_inv; i < state.MaxOrderSize_Limit; i++) {
							rng.genUniform();
						}
					}
				}
			}

			return {demand, arrival_prob };
		}

		double MDP::ModifyStateWithEvent(State& state,const MDP::Event& event) const
		{
			state.cat = StateCategory::AwaitAction();
			int64_t action_num = 0;
			int64_t orders_received = state.orders_received;

			if (state.min_leadtime == 0 && state.state_vector.back() > 0) {
				if (state.stochasticLeadtimes) {
					int64_t inv = state.state_vector.back();
					for (action_num = 0; action_num < inv; action_num++) {
						if (event.second[action_num] <= state.cumulative_leadtime_probs[0]) {
							state.state_vector.front()++;
							state.state_vector.back()--;
							if (state.censoredLeadtime) {
								state.past_leadtimes.front()++;
								state.orders_received++;
							}
						}
					}
					if (state.censoredDemand && state.order_initializationPhase > 0 && state.orders_received > orders_received) {
						state.demand_waits_order = false;
					}
				}
				else {
					state.state_vector.front() += state.state_vector.back();
					state.state_vector.back() = 0;
				}
			}

			int64_t onHand = state.state_vector.pop_front();
			int64_t demand = event.first;
			double cost{ 0.0 };
			bool uncensored = true;
			if (onHand >= demand)
			{
				onHand -= demand;
				state.total_inv -= demand;
				cost = onHand * h;

				if (state.censoredDemand) {
					cost -= demand * state.p;
				}
			}
			else
			{
				state.total_inv -= onHand;

				if (state.censoredDemand) {
					uncensored = false;
					cost -= onHand * state.p;
					demand = onHand;
				} 
				else {
					cost = (demand - onHand) * state.p;
				}

				onHand = 0;
			}

			if (state.max_leadtime > 0) {
				if (state.stochasticLeadtimes) {
					for (int64_t i = std::max((int64_t)1, state.min_leadtime); i <= state.max_leadtime; i++) {
						int64_t ub = state.state_vector.at(max_leadtime - i) + action_num;
						int64_t lb = action_num;
						for (action_num = lb; action_num < ub; action_num++) {
							if (event.second[action_num] <= state.cumulative_leadtime_probs[i]) {
								onHand++;
								state.state_vector.at(max_leadtime - i)--;
								if (state.censoredLeadtime) {
									state.past_leadtimes[i]++;
									state.orders_received++;
								}
							}
						}
					}
				}
				else {
					onHand += state.state_vector.at(max_leadtime - state.max_leadtime);
					state.state_vector.at(max_leadtime - state.max_leadtime) = 0;
				}
			}
			state.state_vector.front() = onHand;

			if (state.collectStatistics) {
				int64_t current_cyclePeriod = state.demand_cycles[state.period];
				if (state.demand_cycles.size() > 1) {
					state.period++;
					state.period = state.period % state.demand_cycles.size();
				}
				if (state.censoredLeadtime && state.orders_received > orders_received)
					UpdateLeadTimeStatistics(state);
				if (state.censoredDemand && (state.order_initializationPhase == 0 || !state.demand_waits_order))
					UpdateDemandStatistics(state, uncensored, demand, current_cyclePeriod);
				if (state.order_initializationPhase == 0) {
					UpdateOrderLimits(state);
				}
				else {
					if (!state.censoredDemand && state.demand_cycles.size() > 1) {
						state.MaxOrderSize = state.cycle_MaxOrderSize[state.period];
						state.MaxSystemInv = state.cycle_MaxSystemInv[state.period];
						state.fractiles = state.cycle_fractiles[state.period];
					}
					state.order_initializationPhase--;
					if (state.censoredDemand && state.order_initializationPhase > 0 && state.orders_received > orders_received)
						state.demand_waits_order = false;
				}
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
		
		void MDP::UpdateOrderLimits(State& state) const {
			std::vector<DiscreteDist> dist_vec_over_leadtime;
			dist_vec_over_leadtime.reserve(state.estimated_max_leadtime - state.estimated_min_leadtime + 1);
			std::vector<DiscreteDist> dist_vec;
			dist_vec.reserve(state.estimated_max_leadtime - state.estimated_min_leadtime + 1);
			for (int64_t j = state.estimated_min_leadtime; j <= state.estimated_max_leadtime; j++)
			{
				auto DemOverLeadtime = DiscreteDist::GetZeroDist();
				for (int64_t k = 0; k < j; k++) {
					int64_t cyclePeriod = (state.period + k) % state.demand_cycles.size();
					DynaPlex::DiscreteDist dist_over_lt = DiscreteDist::GetCustomDist(state.cycle_probs[cyclePeriod], state.cycle_min_demand[cyclePeriod]);
					DemOverLeadtime = DemOverLeadtime.Add(dist_over_lt);
				}
				int64_t cyclePeriod_on_leadtime = state.demand_cycles[(state.period + j) % state.demand_cycles.size()];
				DynaPlex::DiscreteDist dist_on_leadtime = DiscreteDist::GetCustomDist(state.cycle_probs[cyclePeriod_on_leadtime], state.cycle_min_demand[cyclePeriod_on_leadtime]);
				DemOverLeadtime = DemOverLeadtime.Add(dist_on_leadtime);
				dist_vec.push_back(dist_on_leadtime);
				dist_vec_over_leadtime.push_back(DemOverLeadtime);
			}
			std::vector<double> probs_vec(state.estimated_leadtime_probs.begin() + state.estimated_min_leadtime, state.estimated_leadtime_probs.begin() + state.estimated_max_leadtime + 1);
			auto DummyDemOverLeadtime = DiscreteDist::MultipleMix(dist_vec_over_leadtime, probs_vec);
			auto DummyDemOnLeadtime = DiscreteDist::MultipleMix(dist_vec, probs_vec);
			state.MaxOrderSize = std::min(DummyDemOnLeadtime.Fractile(state.p / (state.p + h)), state.MaxOrderSize_Limit);
			state.MaxSystemInv = std::min(DummyDemOverLeadtime.Fractile(state.p / (state.p + h)), MaxSystemInv);
			for (size_t i = 0; i < fractiles.size(); i++) {
				state.fractiles[i] = DummyDemOnLeadtime.Fractile(fractiles[i]);
			}
		}

		void MDP::UpdateLeadTimeStatistics(State& state) const {
			for (int64_t i = 0; i <= max_leadtime; i++) {
				state.estimated_leadtime_probs[i] = state.past_leadtimes[i] / state.orders_received;
			}
			bool found_min = false;
			double total_prob = 0.0;
			for (int64_t i = 0; i <= max_leadtime; i++) {
				double prob = state.estimated_leadtime_probs[i];
				total_prob += prob;
				if (!found_min && prob > 0.0) {
					state.estimated_min_leadtime = i;
					found_min = true;
				}
				if (total_prob == 1.0) {
					state.estimated_max_leadtime = i;
					break;
				}
			}
		}

		void MDP::UpdateDemandStatistics(State& state, bool uncensored, int64_t newObs, int64_t current_cyclePeriod) const {
			state.periodCount[current_cyclePeriod]++;
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
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			features.Add(state.p);
			features.Add(state.state_vector);
			//std::vector<double> dummy_prob_vec(max_leadtime + 1, 1.0);
			//double total_prob = 1.0;
			//for (int64_t i = 0; i <= state.estimated_max_leadtime; i++) {
			//	dummy_prob_vec[i] = state.estimated_leadtime_probs[i] / total_prob;
			//	total_prob -= state.estimated_leadtime_probs[i];
			//}
			for (int64_t i = max_leadtime; i >= 0; i--) {
				features.Add(state.estimated_leadtime_probs[i]);
			}
			//features.Add(state.estimated_leadtime_probs);
			features.Add(state.fractiles);
			for (int64_t i = 0; i <= max_leadtime; i++) {
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
			state.censoredDemand = false;
			state.censoredLeadtime = false;
			state.demand_waits_order = false;
			std::vector<double> mean_true_demand;
			std::vector<double> stdev_true_demand;

			if (evaluate) {
				state.demand_cycles = demand_cycles;
				mean_true_demand = mean_demand;
				stdev_true_demand = stdDemand;
				state.mean_cycle_demand = mean_demand;
				state.std_cycle_demand = stdDemand;
				state.leadtime_probs = leadtime_probs;
				bool found_min = false;
				double total_prob = 0.0;
				for (int64_t i = 0; i <= max_leadtime; i++) {
					double prob = state.leadtime_probs[i];
					total_prob += prob;
					if (!found_min && prob > 0.0) {
						state.min_leadtime = i;
						found_min = true;
					}
					if (std::abs(total_prob - 1.0) < 1e-8) {
						state.max_leadtime = i;
						break;
					}
				}
				state.estimated_leadtime_probs = state.leadtime_probs;
				state.estimated_max_leadtime = state.max_leadtime;
				state.estimated_min_leadtime = state.min_leadtime;
				state.p = p;

				if (collectStatistics) {
					if (censoredDemand) {
						state.periodCount.reserve(state.demand_cycles.size());
						state.past_demands.reserve(state.demand_cycles.size());
						state.cumulative_demands.reserve(state.demand_cycles.size());
						state.censor_indicator.reserve(state.demand_cycles.size());

						for (int64_t i = 0; i < state.demand_cycles.size(); i++) {
							state.mean_cycle_demand[i] = 0.0;
							state.std_cycle_demand[i] = 0.0;
							state.periodCount.push_back(0);
							state.past_demands.push_back({});
							state.cumulative_demands.push_back({});
							state.censor_indicator.push_back({});
							state.censoredDemand = true;
							state.demand_waits_order = true;
						}
					}

					if (censoredLeadtime) {
						state.past_leadtimes.reserve(max_leadtime + 1);
						state.estimated_min_leadtime = max_leadtime;
						state.estimated_max_leadtime = max_leadtime;
						for (int64_t i = 0; i <= max_leadtime; i++) {
							state.estimated_leadtime_probs[i] = 0.0;
							state.past_leadtimes.push_back(0);
						}
						state.estimated_leadtime_probs[state.estimated_max_leadtime] = 1.0;
						state.censoredLeadtime = true;
						state.orders_received = 0;
					}

					if (censoredDemand || censoredLeadtime) {
						state.collectStatistics = true;
						state.order_initializationPhase = std::max(state.estimated_max_leadtime, (int64_t)state.demand_cycles.size());
					}
				}
			}
			else {
				state.p = rng.genUniform() * (max_p - min_p) + min_p;
				state.min_leadtime = static_cast<int64_t>(std::floor(rng.genUniform() * (max_leadtime - min_leadtime + 1))) + min_leadtime;
				state.max_leadtime = static_cast<int64_t>(std::floor(rng.genUniform() * (max_leadtime - state.min_leadtime + 1))) + state.min_leadtime;
				//state.max_leadtime = state.min_leadtime;
				state.leadtime_probs = SampleLeadTimeDistribution(rng, state.min_leadtime, state.max_leadtime);
				state.estimated_leadtime_probs = state.leadtime_probs;
				state.estimated_max_leadtime = state.max_leadtime;
				state.estimated_min_leadtime = state.min_leadtime;
				int64_t num_cycles = static_cast<int64_t>(std::floor(rng.genUniform() * max_num_cycles)) + 1;
				mean_true_demand.reserve(num_cycles);
				stdev_true_demand.reserve(num_cycles);
				state.demand_cycles.reserve(num_cycles);
				state.periodCount.reserve(num_cycles);
				for (int64_t i = 0; i < num_cycles; i++) {
					state.demand_cycles.push_back(i);
					state.periodCount.push_back(0);
					double mean = rng.genUniform() * (max_demand - min_demand) + min_demand;
					mean_true_demand.push_back(mean);
					int64_t n = static_cast<int64_t>(std::round(mean / 0.2));
					double prob = mean / n;
					double binom_stdev = std::sqrt(n * prob * (1.0 - prob));
					stdev_true_demand.push_back(rng.genUniform() * (mean * 2.0 - binom_stdev) + binom_stdev);
				}
				state.mean_cycle_demand = mean_true_demand;
				state.std_cycle_demand = stdev_true_demand;
				state.limitingPeriod = std::max(state.max_leadtime, (int64_t)state.demand_cycles.size());
			}

			if (state.min_leadtime == state.max_leadtime) {
				state.stochasticLeadtimes = false;
			}
			else {
				state.stochasticLeadtimes = true;
				std::vector<double> dummy_prob_vec(max_leadtime + 1, 1.0);
				state.cumulative_leadtime_probs = dummy_prob_vec;
				double total_prob_v1 = 1.0;
				for (int64_t i = 0; i < state.max_leadtime; i++) {
					state.cumulative_leadtime_probs[i] = state.leadtime_probs[i] / total_prob_v1;
					total_prob_v1 -= state.leadtime_probs[i];
				}
			}

			auto queue = Queue<int64_t>{};
			queue.reserve(max_leadtime + 1);
			queue.push_back(0);
			for (int64_t i = 0; i < max_leadtime - 1; i++)
			{
				queue.push_back(0);
			}
			state.cat = StateCategory::AwaitAction();
			state.state_vector = queue;
			state.total_inv = queue.sum();

			state.true_demand_probs.reserve(state.demand_cycles.size());
			state.min_true_demand.reserve(state.demand_cycles.size());
			state.max_true_demand.reserve(state.demand_cycles.size());
			state.cycle_probs.reserve(state.demand_cycles.size());
			state.cycle_min_demand.reserve(state.demand_cycles.size());		
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
				if (!state.censoredDemand) {
					state.cycle_probs.push_back(probs);
					state.cycle_min_demand.push_back(state_demand_dist.Min());
				}
				else {
					state.cycle_probs.push_back({});
					state.cycle_min_demand.push_back(0);
				}
			}

			state.MaxOrderSize_Limit = 0;
			std::vector<double> probs_vec(state.estimated_leadtime_probs.begin() + state.estimated_min_leadtime, state.estimated_leadtime_probs.begin() + state.estimated_max_leadtime + 1);
			double total_prob = 0.0;
			for (int64_t i = 0; i < probs_vec.size(); i++) {
				total_prob += probs_vec[i];
			}
			if (std::abs(total_prob - 1.0) >= 1e-8)
				throw DynaPlex::Error("Initiate state: total lead time probabilities should sum up to 1.0.");

			if (!state.censoredDemand) {
				state.cycle_fractiles.reserve(state.demand_cycles.size());
				state.cycle_MaxOrderSize.reserve(state.demand_cycles.size());
				state.cycle_MaxSystemInv.reserve(state.demand_cycles.size());
				for (int64_t i = 0; i < state.demand_cycles.size(); i++) {
					std::vector<DiscreteDist> dist_vec_over_leadtime;
					dist_vec_over_leadtime.reserve(state.estimated_max_leadtime - state.estimated_min_leadtime + 1);
					std::vector<DiscreteDist> dist_vec;
					dist_vec.reserve(state.estimated_max_leadtime - state.estimated_min_leadtime + 1);
					for (int64_t j = state.estimated_min_leadtime; j <= state.estimated_max_leadtime; j++)
					{	
						auto DemOverLeadtime = DiscreteDist::GetZeroDist();
						for (int64_t k = 0; k < j; k++) {
							int64_t cyclePeriod = (i + k) % state.demand_cycles.size();
							DynaPlex::DiscreteDist dist_over_lt = DiscreteDist::GetCustomDist(state.true_demand_probs[cyclePeriod], state.min_true_demand[cyclePeriod]);
							DemOverLeadtime = DemOverLeadtime.Add(dist_over_lt);
						}
						int64_t cyclePeriod_on_leadtime = (i + j) % state.demand_cycles.size();
						DynaPlex::DiscreteDist dist_on_leadtime = DiscreteDist::GetCustomDist(state.true_demand_probs[cyclePeriod_on_leadtime], state.min_true_demand[cyclePeriod_on_leadtime]);
						DemOverLeadtime = DemOverLeadtime.Add(dist_on_leadtime);
						dist_vec.push_back(dist_on_leadtime);
						dist_vec_over_leadtime.push_back(DemOverLeadtime);
					}
					auto DummyDemOverLeadtime = DiscreteDist::MultipleMix(dist_vec_over_leadtime, probs_vec);
					auto DummyDemOnLeadtime = DiscreteDist::MultipleMix(dist_vec, probs_vec);
					state.MaxOrderSize_Limit = std::max(state.MaxOrderSize_Limit, DummyDemOnLeadtime.Fractile(state.p / (state.p + h)));
					state.cycle_MaxOrderSize.push_back(std::min(MaxOrderSize, DummyDemOnLeadtime.Fractile(state.p / (state.p + h))));
					state.cycle_MaxSystemInv.push_back(std::min(MaxSystemInv, DummyDemOverLeadtime.Fractile(state.p / (state.p + h))));
					std::vector<int64_t> _fractiles(fractiles.size(), 0);
					for (size_t i = 0; i < fractiles.size(); i++) {
						_fractiles[i] = DummyDemOnLeadtime.Fractile(fractiles[i]);
					}
					state.cycle_fractiles.push_back(_fractiles);
				}
				state.MaxOrderSize = state.cycle_MaxOrderSize[state.period];
				state.MaxSystemInv = state.cycle_MaxSystemInv[state.period];
				state.fractiles = state.cycle_fractiles[state.period];
				state.MaxOrderSize_Limit = std::min(MaxOrderSize, state.MaxOrderSize_Limit);
			}
			else {
				DynaPlex::DiscreteDist edge_dist = DiscreteDist::GetAdanEenigeResingDist(max_demand, max_demand * 2);
				std::vector<DiscreteDist> dist_vec_over_leadtime;
				dist_vec_over_leadtime.reserve(state.estimated_max_leadtime - state.estimated_min_leadtime + 1);
				std::vector<DiscreteDist> dist_vec;
				dist_vec.reserve(state.estimated_max_leadtime - state.estimated_min_leadtime + 1);
				for (int64_t j = state.estimated_min_leadtime; j <= state.estimated_max_leadtime; j++)
				{
					auto DemOverLeadtime = DiscreteDist::GetZeroDist();
					for (int64_t k = 0; k < j; k++) {
						DemOverLeadtime = DemOverLeadtime.Add(edge_dist);
					}
					DemOverLeadtime = DemOverLeadtime.Add(edge_dist);
					dist_vec.push_back(edge_dist);
					dist_vec_over_leadtime.push_back(DemOverLeadtime);
				}
				auto DummyDemOverLeadtime = DiscreteDist::MultipleMix(dist_vec_over_leadtime, probs_vec);
				auto DummyDemOnLeadtime = DiscreteDist::MultipleMix(dist_vec, probs_vec);
				state.MaxOrderSize = std::min(MaxOrderSize, DummyDemOnLeadtime.Fractile(state.p / (state.p + h)));
				state.MaxSystemInv = std::min(MaxSystemInv, DummyDemOverLeadtime.Fractile(state.p / (state.p + h)));
				std::vector<int64_t> _fractiles(fractiles.size(), 0);
				state.fractiles = _fractiles;
				for (size_t i = 0; i < fractiles.size(); i++) {
					state.fractiles[i] = DummyDemOnLeadtime.Fractile(fractiles[i]);
				}
				state.MaxOrderSize_Limit = state.MaxOrderSize;		
			}

			return state;
		}

		std::vector<double> MDP::SampleLeadTimeDistribution(RNG& rng, int64_t min_lt, int64_t max_lt) const {
			int64_t possible_leadtimes = max_lt - min_lt + 1;
			std::vector<double> dummy_leadtime_probs(max_leadtime + 1, 0.0);

			if (possible_leadtimes == 1) {
				dummy_leadtime_probs[min_lt] = 1.0;
			}
			else if (rng.genUniform() < 0.2) {
				for (int64_t i = min_lt; i <= max_lt; i++) {
					dummy_leadtime_probs[i] = 1.0 / possible_leadtimes;
				}
			}
			else {
				double mean = rng.genUniform() * (max_lt - min_lt) + min_lt;
				double total_probs = 0.0;
				if (mean > 2.0) {
					int64_t n = static_cast<int64_t>(std::round(mean / 0.2));
					double prob = mean / n;
					double binom_stdev = std::sqrt(n * prob * (1.0 - prob));
					double stdev = rng.genUniform() * (mean * 2.0 - binom_stdev) + binom_stdev;

					DynaPlex::DiscreteDist dist = DiscreteDist::GetAdanEenigeResingDist(mean, stdev);
					for (int64_t i = min_lt; i <= max_lt; i++) {
						double lt_prob = dist.ProbabilityAt(i);
						dummy_leadtime_probs[i] = lt_prob;
						total_probs += lt_prob;
					}
					double remaining_probs = 1.0 - total_probs;
					for (int64_t i = min_lt; i <= max_lt; i++) {
						dummy_leadtime_probs[i] += remaining_probs / possible_leadtimes;
					}
				}
				else {
					for (int64_t i = min_lt; i < max_lt; i++) {
						double lt_prob = rng.genUniform() * (1.0 - total_probs);
						dummy_leadtime_probs[i] = lt_prob;
						total_probs += lt_prob;
					}
					dummy_leadtime_probs[max_lt] = 1.0 - total_probs;
				}
			}

			double total_prob_v2 = 0.0;
			for (int64_t i = 0; i < dummy_leadtime_probs.size(); i++) {
				total_prob_v2 += dummy_leadtime_probs[i];
			}
			if (std::abs(total_prob_v2 - 1.0) >= 1e-8)
				throw DynaPlex::Error("When sampling, total lead time probabilities should sum up to 1.0.");

			return dummy_leadtime_probs;
		}

		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("state_vector", state.state_vector);
			vars.Get("demand_cycles", state.demand_cycles);
			vars.Get("mean_cycle_demand", state.mean_cycle_demand);
			vars.Get("std_cycle_demand", state.std_cycle_demand);
			vars.Get("fractiles", state.fractiles);
			vars.Get("estimated_leadtime_probs", state.estimated_leadtime_probs);
			vars.Get("collectStatistics", state.collectStatistics);
			vars.Get("stochasticLeadtimes", state.stochasticLeadtimes);
			vars.Get("censoredDemand", state.censoredDemand);
			vars.Get("p", state.p);

			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("state_vector", state_vector);
			vars.Add("demand_cycles", demand_cycles);
			vars.Add("mean_cycle_demand", mean_cycle_demand);
			vars.Add("std_cycle_demand", std_cycle_demand);
			vars.Add("fractiles", fractiles);
			vars.Add("estimated_leadtime_probs", estimated_leadtime_probs);
			vars.Add("collectStatistics", collectStatistics);
			vars.Add("stochasticLeadtimes", stochasticLeadtimes);
			vars.Add("censoredDemand", censoredDemand);
			vars.Add("p", p);

			return vars;
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "lost_sales_all",
				/*description*/ "Lost sales problem with cyclic censored demand and stochastic lead times.)",
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

