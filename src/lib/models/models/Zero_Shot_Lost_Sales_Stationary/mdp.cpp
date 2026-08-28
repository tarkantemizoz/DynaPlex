#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>
#include <algorithm>

namespace DynaPlex::Models {
	namespace Zero_Shot_Lost_Sales_Stationary
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
			config.Get("max_leadtime", max_leadtime);
			config.Get("max_demand", max_demand);
			config.Get("max_p", max_p);
			h = 1.0;
			min_p = 2.0;
			min_leadtime = 0;
			min_demand = 2.0;
			censoredDemand = false;
			p = min_p;
			leadtime = min_leadtime;
			mean_demand = min_demand;
			stdDemand = 0.0;

			if (evaluate) {
				if (config.HasKey("censoredDemand"))
					config.Get("censoredDemand", censoredDemand);

				config.Get("p", p);
				config.Get("leadtime", leadtime);
				if (leadtime > max_leadtime || leadtime < min_leadtime)
					throw DynaPlex::Error("MDP instance: Leadtime should be between max_leadtime and min_leadtime.");
				config.Get("mean_demand", mean_demand);
				config.Get("stdDemand", stdDemand);
			}
			else if (config.HasKey("censoredDemand")) {
				// Censoring is a deployment-time (evaluate) mechanism: training always uses the
				// true sampled parameters. Enforce the invariant !evaluate => !censoredDemand
				// rather than silently ignoring a contradictory request.
				bool requested_censoring = false;
				config.Get("censoredDemand", requested_censoring);
				if (requested_censoring)
					throw DynaPlex::Error("MDP instance: censoredDemand is only supported in evaluate mode; set evaluate=true or remove censoredDemand.");
			}

			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;

			DynaPlex::DiscreteDist dist = DiscreteDist::GetAdanEenigeResingDist(max_demand, max_demand * 2);
			//Global caps computed from the worst-case (max demand, max lead time, max penalty) instance:
			auto DemOverLeadtime = DiscreteDist::GetZeroDist();
			for (int64_t i = 0; i <= max_leadtime; i++)
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
			// Use binary search on the cumulative PMF of the (stationary) true demand.
			auto it = std::lower_bound(state.cumulativePMF.begin(), state.cumulativePMF.end(), randomValue);
			size_t index = std::distance(state.cumulativePMF.begin(), it);
			return state.min_true_demand + static_cast<int64_t>(index);
		}

		double MDP::ModifyStateWithEvent(State& state, const MDP::Event& event) const
		{
			state.cat = StateCategory::AwaitAction();
			int64_t onHand = state.state_vector.pop_front();
			int64_t new_coming_orders = 0;

			// Deterministic lead time: the order placed L periods ago arrives now. It sits at
			// pipeline slot (max_leadtime - L); slot (max_leadtime - 1) holds the order just placed,
			// which is the arriving one when L == 0 (or L == 1, arriving next period).
			int64_t loc = (state.leadtime == 0 ? 1 : state.leadtime);
			int64_t& expected = state.state_vector.at(max_leadtime - loc);
			if (expected > 0) {
				if (state.leadtime == 0)
					onHand += expected;
				else
					new_coming_orders += expected;
				expected = 0;
			}

			if (censoredDemand) {
				if (!state.collectDemandStatistics && onHand > 0)
					state.collectDemandStatistics = true;
			}

			int64_t demand = event;
			state.demand = demand;

			// Under censored demand the true demand - and therefore the true lost-sales cost - is
			// unobservable during a stockout. Following the paper, we then optimise an equivalent
			// objective built only from observables (units sold and ending inventory): the sales
			// profit surrogate. It equals (true cost - p * demand); since p * demand is exogenous
			// and policy-independent, minimising it is equivalent to minimising the true cost, while
			// remaining computable when a stockout hides the true demand.
			double cost = 0.0;
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
				rewards = -onHand * state.p; // -p * (units sold); ending inventory is 0

				if (evaluate)
					state.cumulativeStockouts += stockouts;

				if (censoredDemand) {
					uncensored = false;
					demand = onHand;
				}

				onHand = 0;
			}
			state.state_vector.front() = onHand + new_coming_orders;

			if (evaluate && state.cumulativeDemands > 0)
				state.ServiceLevel = static_cast<double>(state.cumulativeDemands - state.cumulativeStockouts) / (static_cast<double>(state.cumulativeDemands));

			// Only under censoring does the demand estimate (and hence the order-up-to limits) evolve;
			// otherwise the demand distribution and lead time are fixed and the limits set in
			// GetInitialState stay valid.
			if (censoredDemand) {
				if (state.collectDemandStatistics)
					UpdateDemandStatistics(state, uncensored, demand); // Call Kaplan - Meier Estimator
				UpdateOrderLimits(state);
			}

			state.OrderConstraint = std::max(static_cast<int64_t>(0), std::min(state.MaxSystemInv - state.total_inv, state.MaxOrderSize));

			// Return the observable sales-profit surrogate under censoring (true cost is unobservable
			// there); otherwise return the exact lost-sales cost.
			return censoredDemand ? rewards : cost;
		}

		std::pair<int64_t, int64_t> MDP::DemandOverLeadtimeFractiles(const State& state, const DiscreteDist& demand_dist) const {
			// Demand-on-leadtime is the single-period demand; demand-over-leadtime is the sum over
			// the L+1 periods of the review cycle (deterministic lead time L).
			auto DemOverLeadtime = DiscreteDist::GetZeroDist();
			for (int64_t k = 0; k <= state.leadtime; k++)
				DemOverLeadtime = DemOverLeadtime.Add(demand_dist);
			double fractile = state.p / (state.p + h);
			return { demand_dist.Fractile(fractile), DemOverLeadtime.Fractile(fractile) };
		}

		void MDP::UpdateOrderLimits(State& state) const {
			DynaPlex::DiscreteDist demand_dist = DiscreteDist::GetCustomDist(state.demand_probs, state.est_min_demand);
			auto [orderSizeFractile, systemInvFractile] = DemandOverLeadtimeFractiles(state, demand_dist);
			state.MaxOrderSize = std::min(orderSizeFractile, state.MaxOrderSize_Limit);
			state.MaxSystemInv = std::min(systemInvFractile, MaxSystemInv);
		}

		void MDP::UpdateDemandStatistics(State& state, bool uncensored, int64_t newObs) const { // Kaplan - Meier Estimator
			state.periodCount++;

			int64_t oldSize = static_cast<int64_t>(state.past_demands.size()) - 1;
			if (newObs > oldSize) {
				for (int64_t i = oldSize + 1; i < newObs; i++) {
					state.past_demands.push_back(0);
					state.censor_indicator.push_back(0);
					state.cumulative_demands.push_back(1);
				}
				state.past_demands.push_back(1);
				state.cumulative_demands.push_back(0);

				if (uncensored)
					state.censor_indicator.push_back(0);
				else
					state.censor_indicator.push_back(1);
			}
			else {
				state.past_demands[newObs]++;
				oldSize = newObs - 1;

				if (!uncensored)
					state.censor_indicator[newObs]++;
			}
			for (int64_t i = 0; i < oldSize + 1; i++) {
				state.cumulative_demands[i]++;
			}

			int64_t demand_size = static_cast<int64_t>(state.past_demands.size());
			std::vector<double> probs(demand_size, 0.0);
			// Iterative weight redistribution
			for (int64_t i = 0; i < demand_size; i++) {
				probs[i] += static_cast<double>(state.past_demands[i]) / state.periodCount;
				if (state.censor_indicator[i] > 0 && i < demand_size - 1) { // Censored observation
					double weight_to_redistribute = static_cast<double>(state.censor_indicator[i]) / state.periodCount;
					probs[i] -= weight_to_redistribute;

					for (int64_t j = i + 1; j < demand_size; ++j) {
						probs[j] += state.past_demands[j] * weight_to_redistribute / state.cumulative_demands[i];
					}
				}
			}
			state.demand_probs = probs;
			state.est_min_demand = 0;
			DynaPlex::DiscreteDist dist = DiscreteDist::GetCustomDist(probs, 0);
			state.mean_demand = dist.Expectation();
			state.std_demand = dist.StandardDeviation();
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
			features.Add(state.p);
			features.Add(state.state_vector);
			features.Add(state.leadtime);
			features.Add(state.mean_demand);
			features.Add(state.std_demand);
		}

		MDP::State MDP::GetInitialState(RNG& rng) const
		{
			State state{};

			state.demand = 0;
			state.collectDemandStatistics = false;
			state.periodCount = 0;
			state.ServiceLevel = 1.0;
			state.cumulativeDemands = 0;
			state.cumulativeStockouts = 0;

			double mean_true_demand;
			double stdev_true_demand;
			DynaPlex::DiscreteDist edge_dist = DiscreteDist::GetConstantDist(static_cast<int64_t>(std::ceil(max_demand)));

			if (evaluate) {
				state.leadtime = leadtime;
				state.p = p;
				mean_true_demand = mean_demand;
				stdev_true_demand = stdDemand;
				state.mean_demand = mean_demand;
				state.std_demand = stdDemand;

				if (censoredDemand) {
					// Expose the pessimistic edge estimate until demand observations arrive.
					state.mean_demand = edge_dist.Expectation();
					state.std_demand = edge_dist.StandardDeviation();
				}
			}
			else {
				state.p = rng.genUniform() * (max_p - min_p) + min_p;
				state.leadtime = static_cast<int64_t>(std::floor(rng.genUniform() * (max_leadtime - min_leadtime + 1))) + min_leadtime;
				double mean = rng.genUniform() * (max_demand - min_demand) + min_demand;
				double min_var = DiscreteDist::LeastVarianceRequiredForAERFit(mean);
				double min_std = std::sqrt(min_var);
				double st_dev = rng.genUniform() * (mean * 2.0 - min_std) + min_std;
				mean_true_demand = mean;
				stdev_true_demand = st_dev;
				state.mean_demand = mean;
				state.std_demand = st_dev;
			}

			// Fixed-size pipeline queue (front is on-hand inventory).
			auto queue = Queue<int64_t>{};
			queue.reserve(max_leadtime + 1);
			queue.push_back(0);
			for (int64_t i = 1; i < max_leadtime; i++)
			{
				queue.push_back(0);
			}
			state.cat = StateCategory::AwaitAction();
			state.state_vector = queue;
			state.total_inv = queue.sum();

			// True (sampling) demand distribution.
			DynaPlex::DiscreteDist state_demand_dist = DiscreteDist::GetAdanEenigeResingDist(mean_true_demand, stdev_true_demand);
			state.min_true_demand = state_demand_dist.Min();
			std::vector<double> cumul_probs;
			cumul_probs.reserve(state_demand_dist.DistinctValueCount());
			double sum = 0.0;
			for (const auto& [qty, prob] : state_demand_dist) {
				sum += prob;
				cumul_probs.push_back(sum);
			}
			state.cumulativePMF = cumul_probs;

			// Demand estimator initialisation.
			if (censoredDemand) {
				// Start pessimistic (constant edge demand) and learn the true distribution online.
				std::vector<double> edge_probs;
				edge_probs.reserve(edge_dist.DistinctValueCount());
				for (const auto& [qty, prob] : edge_dist)
					edge_probs.push_back(prob);
				state.demand_probs = edge_probs;
				state.est_min_demand = edge_dist.Min();
			}
			else {
				std::vector<double> probs;
				probs.reserve(state_demand_dist.DistinctValueCount());
				for (const auto& [qty, prob] : state_demand_dist)
					probs.push_back(prob);
				state.demand_probs = probs;
				state.est_min_demand = state.min_true_demand;
			}

			// Order-up-to limits. The true-demand limits (uncensored) also serve as the ceiling
			// (MaxOrderSize_Limit) that the censored estimate may grow towards.
			double fractile = state.p / (state.p + h);
			auto TrueDemOverLeadtime = DiscreteDist::GetZeroDist();
			for (int64_t k = 0; k <= state.leadtime; k++)
				TrueDemOverLeadtime = TrueDemOverLeadtime.Add(state_demand_dist);
			int64_t trueOrderSize = std::min(state_demand_dist.Fractile(fractile), MaxOrderSize);
			int64_t trueSystemInv = std::min(TrueDemOverLeadtime.Fractile(fractile), MaxSystemInv);
			state.MaxOrderSize_Limit = censoredDemand ? MaxOrderSize : trueOrderSize;

			if (censoredDemand) {
				// Base the initial limits on the (pessimistic edge) estimate.
				UpdateOrderLimits(state);
			}
			else {
				state.MaxOrderSize = trueOrderSize;
				state.MaxSystemInv = trueSystemInv;
			}
			state.OrderConstraint = std::max(static_cast<int64_t>(0), std::min(state.MaxSystemInv - state.total_inv, state.MaxOrderSize));

			return state;
		}

		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("p", state.p);
			vars.Get("state_vector", state.state_vector);
			vars.Get("mean_demand", state.mean_demand);
			vars.Get("std_demand", state.std_demand);
			vars.Get("demand", state.demand);
			vars.Get("leadtime", state.leadtime);

			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("p", p);
			vars.Add("state_vector", state_vector);
			vars.Add("mean_demand", mean_demand);
			vars.Add("std_demand", std_demand);
			vars.Add("demand", demand);
			vars.Add("leadtime", leadtime);
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
				/*=id though which the MDP will be retrievable*/ "Zero_Shot_Lost_Sales_Stationary",
				/*description*/ "Lost sales Super-MDP with stationary (non-cyclic) censored demand and deterministic lead time.",
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
