#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace Zero_Shot_Lost_Sales_Inventory_Control 
	{
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
			config.Get("train_stochastic_leadtimes", train_stochastic_leadtimes);
			config.Get("train_cyclic_demand", train_cyclic_demand);
			config.Get("max_leadtime", max_leadtime);
			config.Get("max_demand", max_demand);
			config.Get("max_p", max_p);
			config.Get("max_num_cycles", max_num_cycles);
			h = 1.0;
			min_p = 2.0;
			min_leadtime = 0;
			min_demand = 2.0;
			maximizeRewards = false;
			
			if (evaluate) {				
				if (config.HasKey("censoredDemand"))
					config.Get("censoredDemand", censoredDemand);
				else
					censoredDemand = false;

				if (config.HasKey("maximizeRewards"))
					config.Get("maximizeRewards", maximizeRewards);
				else
					maximizeRewards = false;

				config.Get("p", p);
				order_crossover = false;
				bool stochastic_leadtime = false;
				if (config.HasKey("stochastic_leadtime"))
					config.Get("stochastic_leadtime", stochastic_leadtime);

				config.Get("demand_cycles", demand_cycles);
				config.Get("mean_demand", mean_demand);
				config.Get("stdDemand", stdDemand);
				if (mean_demand.size() != demand_cycles.size() || stdDemand.size() != demand_cycles.size())
					throw DynaPlex::Error("MDP instance: Size of mean/std demand should be equal to demand cycle size.");

				if (!stochastic_leadtime) {
					std::vector<double> probs(max_leadtime + 1, 0.0);
					leadtime_probs = probs;
					censoredLeadtime = false;
					int64_t leadtime;
					config.Get("leadtime", leadtime);
					if (leadtime > max_leadtime || leadtime < min_leadtime)
						throw DynaPlex::Error("MDP instance: Leadtime should be between max_leadtime and min_leadtime.");
					else
						leadtime_probs[leadtime] = 1.0;
				}
				else if (config.HasKey("leadtime_distribution")) {
					stochastic_leadtime = true;
					if (config.HasKey("censoredLeadtime"))
						config.Get("censoredLeadtime", censoredLeadtime);
					else
						censoredLeadtime = false;
					if (config.HasKey("order_crossover"))
						config.Get("order_crossover", order_crossover);

					if (order_crossover) {
						config.Get("leadtime_distribution", leadtime_probs);
						if (leadtime_probs.size() != max_leadtime + 1)
							throw DynaPlex::Error("MDP instance: Size of leadtime probability vector should be max_leadtime + 1.");
					}
					else {
						config.Get("leadtime_distribution", non_crossing_leadtime_rv_probs);
						if (non_crossing_leadtime_rv_probs.size() != max_leadtime + 1)
							throw DynaPlex::Error("MDP instance: Size of non_crossing_leadtime_rv_probs vector should be max_leadtime + 1.");
						double total_prob_v2 = 0.0;
						for (const auto& prob : non_crossing_leadtime_rv_probs)
						{
							if (prob < 0.0)
								throw DynaPlex::Error("MDP instance: non-crossover lead time probability is negative.");
							total_prob_v2 += prob;
						}
						if (std::abs(total_prob_v2 - 1.0) >= 1e-8)
							throw DynaPlex::Error("MDP instance: non-crossover total lead time probability should be 1.0.");

						std::vector<double> cumul_probs(max_leadtime + 1, 0.0);
						double total_prob_v1 = 0.0;
						for (int64_t i = 0; i <= max_leadtime; i++) {
							cumul_probs[i] = non_crossing_leadtime_rv_probs[i] + total_prob_v1;
							total_prob_v1 += non_crossing_leadtime_rv_probs[i];
						}
						std::vector<double> probs(max_leadtime + 1, 0.0);
						leadtime_probs = probs;
						double total_probs = 0.0;
						for (int64_t i = 0; i <= max_leadtime; i++) {
							double prob = 1.0;
							for (int64_t j = 0; j < i; j++) {
								prob *= std::max(0.0, 1.0 - cumul_probs[j]);
							}
							prob *= std::max(0.0, cumul_probs[i]);
							leadtime_probs[i] = prob;
							total_probs += prob;
						}
						// Normalize the probabilities if the total differs from 1
						if (std::abs(total_probs - 1.0) >= 1e-8) {
							for (double& prob : leadtime_probs) {
								prob /= total_probs;
							}
						}
					}
					double total_probs = 0.0;
					for (const auto& prob : leadtime_probs)
					{
						if (prob < 0.0)
							throw DynaPlex::Error("MDP instance: lead time probability is negative.");
						total_probs += prob;
					}
					if (std::abs(total_probs - 1.0) >= 1e-8)
						throw DynaPlex::Error("MDP instance: total lead time probability should be 1.0.");
				}
				else {
					throw DynaPlex::Error("MDP instance: Provide a leadtime value or leadtime distribution.");
				}

			}

			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;

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
				return action <= state.OrderConstraint;
		}

		MDP::Event MDP::GetEvent(const State& state, RNG& rng) const {
			double randomValue = rng.genUniform();
			// Use binary search on the cumulativePMF
			auto it = std::lower_bound(state.cumulativePMFs[state.period].begin(), state.cumulativePMFs[state.period].end(), randomValue);
			size_t index = std::distance(state.cumulativePMFs[state.period].begin(), it);
			int64_t demand = state.min_true_demand[state.period] + static_cast<int64_t>(index);

			if (state.stochasticLeadtimes) {
				std::vector<double> arrival_prob{};
				if (state.order_crossover) {
					int64_t last_order = state.state_vector.back();
					int64_t min_positive_leadtime = std::max((int64_t)1, state.min_leadtime);
					if (state.min_leadtime == 0) {
						for (int64_t i = 0; i < last_order; i++) {
							arrival_prob.push_back(rng.genUniform());
						}
						for (int64_t i = last_order; i < state.MaxOrderSize_Limit; i++) {
							rng.genUniform();
						}
					}
					for (int64_t j = min_positive_leadtime; j < state.max_leadtime; j++) {
						int64_t inv = state.state_vector.at(max_leadtime + 1 - j);
						for (int64_t i = 0; i < inv; i++) {
							arrival_prob.push_back(rng.genUniform());
						}
						for (int64_t i = inv; i < state.MaxOrderSize_Limit; i++) {
							rng.genUniform();
						}
					}
				}
				else {
					arrival_prob.push_back(rng.genUniform());
				}
				return { demand, arrival_prob };
			}
			else {
				return { demand, {} };
			}
		}

		double MDP::ModifyStateWithEvent(State& state, const MDP::Event& event) const
		{
			state.cat = StateCategory::AwaitAction();
			int64_t orders_received = state.orders_received;
			int64_t onHand = state.state_vector.pop_front();
			int64_t new_coming_orders = 0;

			if (state.stochasticLeadtimes) {
				// Same dynamics for training and evaluation; the censoredLeadtime blocks only
				// collect estimator statistics and are inert during training (censoredLeadtime == false).
				if (state.order_crossover) { // order crossover
					int64_t action_num = 0;
					int64_t last_order = state.state_vector.back();
					int64_t min_positive_leadtime = std::max((int64_t)1, state.min_leadtime);
					if (state.min_leadtime == 0 && last_order > 0) {
						const double prob = state.cumulative_leadtime_probs[0];
						int64_t decrement_count = std::count_if(event.second.begin(), event.second.begin() + last_order,
							[prob](double value) { return value <= prob; });
						state.state_vector.back() -= decrement_count;
						onHand += decrement_count;
						action_num = last_order;
						if (state.censoredLeadtime) {
							state.past_leadtimes[0] += decrement_count;
							state.orders_received += decrement_count;
						}
					}
					for (int64_t i = min_positive_leadtime; i < state.max_leadtime; i++) {
						int64_t& current_expected = state.state_vector.at(max_leadtime - i);
						if (current_expected > 0) {
							int64_t lb = action_num;
							int64_t ub = current_expected + lb;
							const double prob = state.cumulative_leadtime_probs[i];
							int64_t decrement_count = std::count_if(event.second.begin() + lb, event.second.begin() + ub,
								[prob](double value) { return value <= prob; });
							current_expected -= decrement_count;
							new_coming_orders += decrement_count;
							action_num = ub;
							if (state.censoredLeadtime) {
								state.past_leadtimes[i] += decrement_count;
								state.orders_received += decrement_count;
							}
						}
					}
					int64_t& last_expected = state.state_vector.at(max_leadtime - state.max_leadtime);
					if (last_expected > 0) {
						new_coming_orders += last_expected;
						if (state.censoredLeadtime) {
							state.past_leadtimes[state.max_leadtime] += last_expected;
							state.orders_received += last_expected;
						}
						last_expected = 0;
					}
				}
				else { // no crossover
					double random_var = event.second.front();
					for (int64_t i = state.min_leadtime; i <= state.max_leadtime; i++) {
						int64_t base_loc = (i == 0 ? 1 : i);
						int64_t& earliest_received = state.state_vector.at(max_leadtime - base_loc);
						if (random_var <= state.cumulative_leadtime_probs[i] && earliest_received > 0) {
							int64_t last_observed = i;
							if (i == 0)
								onHand += earliest_received;
							else
								new_coming_orders += earliest_received;
							earliest_received = 0;
							for (int64_t j = i + 1; j <= state.max_leadtime; j++) {
								int64_t& received = state.state_vector.at(max_leadtime - j);
								if (received > 0) {
									last_observed = j;
									new_coming_orders += received;
									received = 0;
								}
							}
							if (state.censoredLeadtime) {
								for (int64_t j = i; j <= last_observed; j++) {
									state.past_leadtimes[j]++;
									state.orders_received++;
								}
							}
							break;
						}
					}
				}
			}
			else { // deterministic leadtime
				int64_t loc = (state.max_leadtime == 0 ? 1 : state.max_leadtime);
				int64_t& expected = state.state_vector.at(max_leadtime - loc);
				if (expected > 0) {
					if (state.max_leadtime == 0)
						onHand += expected;
					else
						new_coming_orders += expected;
					expected = 0;
				}
			}

			if (state.censoredDemand) {
				if (!state.collectDemandStatistics[state.period] && onHand > 0)
					state.collectDemandStatistics[state.period] = true;
			}				

			int64_t demand = event.first;
			state.demand = demand;

			double cost =  0.0;
			double rewards = 0.0;
			bool uncensored = true;
			if (evaluate) 
				state.cumulativeDemands += demand;

			if (onHand >= demand)
			{
				onHand -= demand;
				state.total_inv -= demand;
				cost = onHand * h;
				rewards = cost - demand * state.p;
			}
			else
			{
				int64_t stockouts = demand - onHand;
				state.total_inv -= onHand;
				cost = stockouts * state.p;
				rewards = -onHand * state.p;

				if (evaluate) 
					state.cumulativeStockouts += stockouts;

				if (state.censoredDemand) {
					uncensored = false;
					demand = onHand;
				} 

				onHand = 0;
			}
			state.state_vector.front() = onHand + new_coming_orders;

			if (evaluate && state.cumulativeDemands > 0)
				state.ServiceLevel = static_cast<double>(state.cumulativeDemands - state.cumulativeStockouts) / (static_cast<double>(state.cumulativeDemands));

			if (state.collectStatistics) {
				if (state.censoredLeadtime && state.orders_received > orders_received)
					UpdateLeadTimeStatistics(state);
				if (state.censoredDemand && state.collectDemandStatistics[state.period])
					UpdateDemandStatistics(state, uncensored, demand); // Call Kaplan - Meier Estimator		
				int64_t old_period = state.period;
				state.period++;
				state.period = state.period % state.cycle_length;
				UpdateOrderLimits(state);
			}
			else {
				state.period++;
				state.period = state.period % state.cycle_length;
				state.MaxOrderSize = state.cycle_MaxOrderSize[state.period];
				state.MaxSystemInv = state.cycle_MaxSystemInv[state.period];
			}

			state.OrderConstraint = std::max(static_cast<int64_t>(0), std::min(state.MaxSystemInv - state.total_inv, state.MaxOrderSize));
			

			if (!maximizeRewards)
				return cost;
			else
				return rewards;
		}

		std::pair<int64_t, int64_t> MDP::DemandOverLeadtimeFractiles(const State& state, int64_t base_period, const std::vector<DiscreteDist>& cycle_demand_dists, const std::vector<double>& leadtime_probs) const {
			int64_t possible_leadtimes = state.estimated_max_leadtime - state.estimated_min_leadtime + 1;
			std::vector<DiscreteDist> dist_vec;
			dist_vec.reserve(possible_leadtimes);
			std::vector<DiscreteDist> dist_vec_over_leadtime;
			dist_vec_over_leadtime.reserve(possible_leadtimes);
			for (int64_t j = state.estimated_min_leadtime; j <= state.estimated_max_leadtime; j++)
			{
				auto DemOverLeadtime = DiscreteDist::GetZeroDist();
				for (int64_t k = 0; k < j; k++) {
					DemOverLeadtime = DemOverLeadtime.Add(cycle_demand_dists[(base_period + k) % state.cycle_length]);
				}
				const DiscreteDist& dist_on_leadtime = cycle_demand_dists[(base_period + j) % state.cycle_length];
				DemOverLeadtime = DemOverLeadtime.Add(dist_on_leadtime);
				dist_vec.push_back(dist_on_leadtime);
				dist_vec_over_leadtime.push_back(DemOverLeadtime);
			}
			auto DummyDemOnLeadtime = DiscreteDist::MultipleMix(dist_vec, leadtime_probs);
			auto DummyDemOverLeadtime = DiscreteDist::MultipleMix(dist_vec_over_leadtime, leadtime_probs);
			double fractile = state.p / (state.p + h);
			return { DummyDemOnLeadtime.Fractile(fractile), DummyDemOverLeadtime.Fractile(fractile) };
		}

		void MDP::UpdateOrderLimits(State& state) const {
			std::vector<double> probs_vec(state.estimated_leadtime_probs.begin() + state.estimated_min_leadtime, state.estimated_leadtime_probs.begin() + state.estimated_max_leadtime + 1);
			std::vector<DiscreteDist> cycle_demand_dists;
			cycle_demand_dists.reserve(state.cycle_length);
			for (int64_t cp = 0; cp < state.cycle_length; cp++)
				cycle_demand_dists.push_back(DiscreteDist::GetCustomDist(state.cycle_probs[cp], state.cycle_min_demand[cp]));
			auto [orderSizeFractile, systemInvFractile] = DemandOverLeadtimeFractiles(state, state.period, cycle_demand_dists, probs_vec);
			state.MaxOrderSize = std::min(orderSizeFractile, state.MaxOrderSize_Limit);
			state.MaxSystemInv = std::min(systemInvFractile, MaxSystemInv);
		}

		void MDP::UpdateLeadTimeStatistics(State& state) const {
			std::vector<int64_t> dummy_past_leadtimes = state.past_leadtimes;
			int64_t dummy_orders_received = state.orders_received;
			if (state.estimated_max_leadtime < max_leadtime) {
				int64_t to_be_received = 0;
				for (int64_t i = std::max((int64_t) 1, state.estimated_max_leadtime); i < max_leadtime; i++) {
					to_be_received += state.state_vector.at(max_leadtime - i);
				}
				if (to_be_received > 0) {
					int64_t leadtimes_inbetween = max_leadtime - state.estimated_max_leadtime;
					int64_t fractional_received = static_cast<int64_t>(std::floor((double) to_be_received / leadtimes_inbetween));
					int64_t total_distributed = 0;
					for (int64_t i = state.estimated_max_leadtime + 1; i < max_leadtime; i++) {
						dummy_past_leadtimes[i] += fractional_received;
						total_distributed += fractional_received;
					}
					dummy_past_leadtimes[max_leadtime] += (to_be_received - total_distributed);
					dummy_orders_received += to_be_received;
				}
			}

			for (int64_t i = 0; i <= max_leadtime; i++) {
				state.estimated_leadtime_probs[i] = static_cast<double>(dummy_past_leadtimes[i]) / static_cast<double>(dummy_orders_received);
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
				if (std::abs(total_prob - 1.0) < 1e-8) {
					state.estimated_max_leadtime = i;
					break;
				}
			}
		} 

		void MDP::UpdateDemandStatistics(State& state, bool uncensored, int64_t newObs) const { // Kaplan - Meier Estimator
			int64_t current_cyclePeriod = state.demand_cycles[state.period];
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
			for (int64_t i = 0; i < demand_size; i++) {
				probs[i] += static_cast<double>(state.past_demands[current_cyclePeriod][i]) / state.periodCount[current_cyclePeriod];
				if (state.censor_indicator[current_cyclePeriod][i] > 0 && i < demand_size - 1) { // Censored observation
					double weight_to_redistribute = static_cast<double>(state.censor_indicator[current_cyclePeriod][i]) / state.periodCount[current_cyclePeriod];
					probs[i] -= weight_to_redistribute;

					for (int64_t j = i + 1; j < demand_size; ++j) {
						probs[j] += state.past_demands[current_cyclePeriod][j] * weight_to_redistribute / state.cumulative_demands[current_cyclePeriod][i];
					}
				}
			}
			for (int64_t i = 0; i < state.cycle_length; i++) {
				if (state.demand_cycles[i] == current_cyclePeriod) {
					state.cycle_probs[i] = probs;
					state.cycle_min_demand[i] = 0;
					DynaPlex::DiscreteDist dist = DiscreteDist::GetCustomDist(probs, 0);
					state.mean_cycle_demand[i] = dist.Expectation();
					state.std_cycle_demand[i] = dist.StandardDeviation();
					state.periodCount[i] = state.periodCount[current_cyclePeriod];
					state.past_demands[i] = state.past_demands[current_cyclePeriod];
					state.censor_indicator[i] = state.censor_indicator[current_cyclePeriod];
					state.cumulative_demands[i] = state.cumulative_demands[current_cyclePeriod];
				}
			}
		}

		std::vector<double> MDP::ReturnUsefulStatistics(const State& state) const
		{
			return { state.ServiceLevel };
		}

		void MDP::ResetHiddenStateVariables(State& state, RNG& rng) const
		{
			state.ServiceLevel = 1.0;
			state.cumulativeDemands = 0;
			state.cumulativeStockouts = 0;
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			if (train_stochastic_leadtimes) {
				if (state.order_crossover)
					features.Add(1);
				else
					features.Add(0);
			}
			features.Add(state.p);
			features.Add(state.state_vector);
			if (train_stochastic_leadtimes) {
				for (int64_t i = max_leadtime; i >= 0; i--) {
					features.Add(state.estimated_leadtime_probs[i]);
				}
			}
			else {
				features.Add(state.min_leadtime);
			}
			if (train_cyclic_demand) {
				features.Add(state.cycle_length);
				for (int64_t i = 0; i < max_num_cycles; i++) {
					int64_t cyclePeriod = (state.period + i) % state.cycle_length;
					features.Add(state.mean_cycle_demand[cyclePeriod]);
					features.Add(state.std_cycle_demand[cyclePeriod]);
				}
			}
			else {
				features.Add(state.mean_cycle_demand.front());
				features.Add(state.std_cycle_demand.front());
			}
		}

		MDP::State MDP::GetInitialState(RNG& rng) const
		{
			State state{};

			state.period = 0;
			state.demand = 0;
			state.collectStatistics = false;
			state.censoredDemand = false;
			state.censoredLeadtime = false;
			std::vector<double> mean_true_demand;
			std::vector<double> stdev_true_demand;
			std::vector<double> leadtime_true_probs;
			DynaPlex::DiscreteDist edge_dist = DiscreteDist::GetConstantDist(static_cast<int64_t>(std::ceil(max_demand)));

			if (evaluate) {
				state.ServiceLevel = 1.0;
				state.cumulativeDemands = 0;
				state.cumulativeStockouts = 0;
				state.demand_cycles = demand_cycles;
				state.cycle_length = demand_cycles.size();
				mean_true_demand = mean_demand;
				stdev_true_demand = stdDemand;
				state.mean_cycle_demand = mean_demand;
				state.std_cycle_demand = stdDemand;
				leadtime_true_probs = leadtime_probs;
				bool found_min = false;
				double total_prob = 0.0;
				for (int64_t i = 0; i <= max_leadtime; i++) {
					double prob = leadtime_true_probs[i];
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
				state.estimated_leadtime_probs = leadtime_true_probs;
				state.estimated_max_leadtime = state.max_leadtime;
				state.estimated_min_leadtime = state.min_leadtime;

				state.p = p;

				if (censoredDemand) {
					state.collectDemandStatistics.reserve(state.cycle_length);
					state.periodCount.reserve(state.cycle_length);
					state.past_demands.reserve(state.cycle_length);
					state.cumulative_demands.reserve(state.cycle_length);
					state.censor_indicator.reserve(state.cycle_length);
					for (int64_t i = 0; i < state.cycle_length; i++) {
						state.collectDemandStatistics.push_back(false);
						state.mean_cycle_demand[i] = edge_dist.Expectation();
						state.std_cycle_demand[i] = edge_dist.StandardDeviation();
						state.periodCount.push_back(0);
						state.past_demands.push_back({});
						state.cumulative_demands.push_back({});
						state.censor_indicator.push_back({});
					}
					state.censoredDemand = true;
					state.collectStatistics = true;
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
					state.collectStatistics = true;
					state.orders_received = 0;
				}			
			}
			else {
				state.p = rng.genUniform() * (max_p - min_p) + min_p;
				state.min_leadtime = static_cast<int64_t>(std::floor(rng.genUniform() * (max_leadtime - min_leadtime + 1))) + min_leadtime;
				if (train_stochastic_leadtimes)
					state.max_leadtime = static_cast<int64_t>(std::floor(rng.genUniform() * (max_leadtime - state.min_leadtime + 1))) + state.min_leadtime;
				else
					state.max_leadtime = state.min_leadtime;
				leadtime_true_probs = SampleLeadTimeDistribution(rng, state.min_leadtime, state.max_leadtime);
				state.estimated_leadtime_probs = leadtime_true_probs;
				state.estimated_max_leadtime = state.max_leadtime;
				state.estimated_min_leadtime = state.min_leadtime;
				state.cycle_length = (int64_t)1;
				if (train_cyclic_demand)
					state.cycle_length += static_cast<int64_t>(std::floor(rng.genUniform() * max_num_cycles));
				mean_true_demand.reserve(state.cycle_length);
				stdev_true_demand.reserve(state.cycle_length);
				state.demand_cycles.reserve(state.cycle_length);
				for (int64_t i = 0; i < state.cycle_length; i++) {
					state.demand_cycles.push_back(i);
					double mean = rng.genUniform() * (max_demand - min_demand) + min_demand;
					mean_true_demand.push_back(mean);
					double min_var = DiscreteDist::LeastVarianceRequiredForAERFit(mean);
					double min_std = std::sqrt(min_var);
					double st_dev = rng.genUniform() * (mean * 2.0 - min_std) + min_std;
					stdev_true_demand.push_back(st_dev);
				}
				state.mean_cycle_demand = mean_true_demand;
				state.std_cycle_demand = stdev_true_demand;
			}

			if (state.min_leadtime == state.max_leadtime) {
				state.stochasticLeadtimes = false;
				state.order_crossover = false;
			}
			else {
				state.stochasticLeadtimes = true;
				if (!evaluate) {
					if (rng.genUniform() < 0.5)
						state.order_crossover = false;
					else 
						state.order_crossover = true;
				}
				else {
					state.order_crossover = order_crossover;
				}
				std::vector<double> dummy_prob_vec(max_leadtime + 1, 1.0);
				state.cumulative_leadtime_probs = dummy_prob_vec;
				if (state.order_crossover) {
					double total_prob_v1 = 1.0;
					for (int64_t i = 0; i < state.max_leadtime; i++) {
						state.cumulative_leadtime_probs[i] = leadtime_true_probs[i] / total_prob_v1;
						total_prob_v1 -= leadtime_true_probs[i];
					}
				}
				else {
					double total_prob_v1 = 0.0;
					if (evaluate) {
						for (int64_t i = 0; i < state.max_leadtime; i++) {
							state.cumulative_leadtime_probs[i] = non_crossing_leadtime_rv_probs[i] + total_prob_v1;
							total_prob_v1 += non_crossing_leadtime_rv_probs[i];
						}
					}
					else {
						for (int64_t i = 0; i < state.max_leadtime; i++) {
							state.cumulative_leadtime_probs[i] = leadtime_true_probs[i] + total_prob_v1;
							total_prob_v1 += leadtime_true_probs[i];
						}
						double total_probs = 0.0;
						for (int64_t i = state.min_leadtime; i <= state.max_leadtime; i++) {
							double prob = 1.0;
							for (int64_t j = state.min_leadtime; j < i; j++) {
								prob *= std::max(0.0, 1.0 - state.cumulative_leadtime_probs[j]);
							}
							prob *= std::max(0.0, state.cumulative_leadtime_probs[i]);
							state.estimated_leadtime_probs[i] = prob;
							total_probs += prob;
						}
						// Normalize the probabilities if the total differs from 1
						if (std::abs(total_probs - 1.0) >= 1e-8) {
							for (double& prob : state.estimated_leadtime_probs) {
								prob /= total_probs;
							}
						}

						double total_prob_v2 = 0.0;
						for (const auto& prob : state.estimated_leadtime_probs)
						{
							if (prob < 0.0)
								throw DynaPlex::Error("Initiate state: non-crossover lead time probability is negative.");
							total_prob_v2 += prob;
						}
						if (std::abs(total_prob_v2 - 1.0) >= 1e-8)
							throw DynaPlex::Error("Initiate state: non-crossover total lead time probability should be 1.0.");
					}
				}
			}

			auto queue = Queue<int64_t>{}; //queue for state vector
			queue.reserve(max_leadtime + 1);
			queue.push_back(0);
			for (int64_t i = 1; i < max_leadtime; i++)
			{
				queue.push_back(0);
			}
			state.cat = StateCategory::AwaitAction();
			state.state_vector = queue;
			state.total_inv = queue.sum();

			std::vector<std::vector<double>> true_demand_probs;
			true_demand_probs.reserve(state.cycle_length);
			state.min_true_demand.reserve(state.cycle_length);
			state.cumulativePMFs.reserve(state.cycle_length);
			for (int64_t i = 0; i < state.cycle_length; i++) {
				DynaPlex::DiscreteDist state_demand_dist = DiscreteDist::GetAdanEenigeResingDist(mean_true_demand[i], stdev_true_demand[i]);
				state.min_true_demand.push_back(state_demand_dist.Min());
				std::vector<double> probs;
				probs.reserve(state_demand_dist.DistinctValueCount());
				std::vector<double> cumul_probs;
				cumul_probs.reserve(state_demand_dist.DistinctValueCount());
				double sum = 0.0;
				for (const auto& [qty, prob] : state_demand_dist) {
					probs.push_back(prob);
					sum += prob;
					cumul_probs.push_back(sum);
				}
				true_demand_probs.push_back(probs);
				state.cumulativePMFs.push_back(cumul_probs);
			}

			if (state.collectStatistics) {
				state.cycle_probs.reserve(state.cycle_length);
				state.cycle_min_demand.reserve(state.cycle_length);
				if (state.censoredDemand) {
					int64_t edge_min = edge_dist.Min();
					std::vector<double> edge_probs;
					edge_probs.reserve(edge_dist.DistinctValueCount());
					for (const auto& [qty, prob] : edge_dist) {
						edge_probs.push_back(prob);
					}
					for (int64_t i = 0; i < state.cycle_length; i++) {
						state.cycle_probs.push_back(edge_probs);
						state.cycle_min_demand.push_back(edge_min);
					}
				}
				else {
					for (int64_t i = 0; i < state.cycle_length; i++) {
						state.cycle_probs.push_back(true_demand_probs[i]);
						state.cycle_min_demand.push_back(state.min_true_demand[i]);
					}
				}
			}

			state.MaxOrderSize_Limit = 0;
			std::vector<double> probs_vec(state.estimated_leadtime_probs.begin() + state.estimated_min_leadtime, state.estimated_leadtime_probs.begin() + state.estimated_max_leadtime + 1);
			state.cycle_MaxOrderSize.reserve(state.cycle_length);
			state.cycle_MaxSystemInv.reserve(state.cycle_length);
			std::vector<DiscreteDist> cycle_demand_dists;
			cycle_demand_dists.reserve(state.cycle_length);
			for (int64_t cp = 0; cp < state.cycle_length; cp++)
				cycle_demand_dists.push_back(state.censoredDemand ? edge_dist : DiscreteDist::GetCustomDist(true_demand_probs[cp], state.min_true_demand[cp]));
			for (int64_t i = 0; i < state.cycle_length; i++) {
				auto [orderSizeFractile, systemInvFractile] = DemandOverLeadtimeFractiles(state, i, cycle_demand_dists, probs_vec);
				int64_t OrderSize = std::min(orderSizeFractile, MaxOrderSize);
				state.MaxOrderSize_Limit = std::max(state.MaxOrderSize_Limit, OrderSize);
				state.cycle_MaxOrderSize.push_back(OrderSize);
				state.cycle_MaxSystemInv.push_back(std::min(MaxSystemInv, systemInvFractile));
			}
			state.MaxOrderSize = state.cycle_MaxOrderSize[state.period];
			state.MaxOrderSize_Limit = state.censoredDemand ? MaxOrderSize : std::min(MaxOrderSize, state.MaxOrderSize_Limit);		
			state.MaxSystemInv = state.cycle_MaxSystemInv[state.period];
			state.OrderConstraint = std::max(static_cast<int64_t>(0), std::min(state.MaxSystemInv - state.total_inv, state.MaxOrderSize));

			return state;
		}

		std::vector<double> MDP::SampleLeadTimeDistribution(RNG& rng, int64_t min_lt, int64_t max_lt) const {
			int64_t possible_leadtimes = max_lt - min_lt + 1;
			std::vector<double> dummy_leadtime_probs(max_leadtime + 1, 0.0);
			std::string dist_type = "none";
			double total_probs = 0.0;

			if (possible_leadtimes == 1) {
				dummy_leadtime_probs[min_lt] = 1.0;
				total_probs = 1.0;
				dist_type = "Deterministic";
			}
			else if (rng.genUniform() < 0.33) {
				double prob = 1.0 / possible_leadtimes;
				for (int64_t i = min_lt; i <= max_lt; i++) {
					dummy_leadtime_probs[i] = prob;
					total_probs += prob;
				}
				dist_type = "Uniform";
			}
			else {
				double mean = (double)(max_lt + min_lt) / 2.0;
				if (rng.genUniform() < 0.5 && mean > 2.0) {
					double min_var = DiscreteDist::LeastVarianceRequiredForAERFit(mean);
					double min_std = std::sqrt(min_var);
					double stdev = rng.genUniform() * (mean * 2.0 - min_std) + min_std;
					DynaPlex::DiscreteDist dist = DiscreteDist::GetAdanEenigeResingDist(mean, stdev);
					for (int64_t i = min_lt; i <= max_lt; i++) {
						double lt_prob = dist.ProbabilityAt(i);
						dummy_leadtime_probs[i] = lt_prob;
						total_probs += lt_prob;
					}
					dist_type = "AER";
				}
				else {
					double remaining_probability = 1.0;
					for (int64_t i = min_lt; i < max_lt; i++) {
						double lt_prob = std::max(0.0, rng.genUniform() * remaining_probability);
						dummy_leadtime_probs[i] = lt_prob;
						total_probs += lt_prob;
						remaining_probability -= lt_prob;
					}
					double lt_prob = std::max(0.0, remaining_probability);
					dummy_leadtime_probs[max_lt] = lt_prob;
					total_probs += lt_prob;
					dist_type = "RAND";
				}
			}
			// Normalize the probabilities if the total differs from 1
			if (std::abs(total_probs - 1.0) >= 1e-8) {
				for (double& prob : dummy_leadtime_probs) {
					prob /= total_probs;
				}
			}
			if (std::any_of(dummy_leadtime_probs.begin(), dummy_leadtime_probs.end(), [](double num) { return std::isnan(num); })) {
				throw DynaPlex::Error("Initiate state: sample lead time: probability value is Nan. Dist: " + dist_type);
			}

			double total_prob_v2 = 0.0;
			for (const auto& prob : dummy_leadtime_probs)
			{
				if (prob < 0.0)
					throw DynaPlex::Error("Initiate state: sample lead time: lead time probability is negative. Dist: " + dist_type);
				total_prob_v2 += prob;
			}
			if (std::abs(total_prob_v2 - 1.0) >= 1e-8)
				throw DynaPlex::Error("Initiate state - sample lead time: total lead time probabilities should sum up to 1.0. Dist: " + dist_type);

			return dummy_leadtime_probs;
		}

		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("p", state.p);
			vars.Get("state_vector", state.state_vector);
			vars.Get("mean_cycle_demand", state.mean_cycle_demand);
			vars.Get("std_cycle_demand", state.std_cycle_demand);
			vars.Get("demand", state.demand);
			if (train_cyclic_demand) {
				vars.Get("period", state.period);
				vars.Get("cycle_length", state.cycle_length);
			}
			if (train_stochastic_leadtimes) {
				vars.Get("order_crossover", state.order_crossover);
				vars.Get("estimated_leadtime_probs", state.estimated_leadtime_probs);
			}
			else {
				vars.Get("min_leadtime", state.min_leadtime);
			}

			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("p", p);
			vars.Add("state_vector", state_vector);
			vars.Add("mean_cycle_demand", mean_cycle_demand);
			vars.Add("std_cycle_demand", std_cycle_demand);
			vars.Add("demand", demand);
			vars.Add("period", period);
			vars.Add("cycle_length", cycle_length);
			vars.Add("order_crossover", order_crossover);
			vars.Add("estimated_leadtime_probs", estimated_leadtime_probs);
			vars.Add("min_leadtime", min_leadtime);
			vars.Add("OrderConstraint", OrderConstraint);

			return vars;
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "Zero_Shot_Lost_Sales_Inventory_Control",
				/*description*/ "Lost sales problem with cyclic censored demand, censored stochastic lead times.)",
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

