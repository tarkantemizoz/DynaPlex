#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"

namespace DynaPlex::Models {
	namespace Zero_Shot_Lost_Sales_Stationary
	{
		// Simplified Super-MDP for lost sales inventory control.
		//
		// This is the "generally capable" (zero-shot) lost sales problem of
		// Zero_Shot_Lost_Sales_Inventory_Control, restricted to the stationary setting:
		//   - the lead time is deterministic within an instance (a single scalar L), but
		//     the Super-MDP randomizes L uniformly over {0,...,max_leadtime} across
		//     training instances (evaluate == false). There is no order crossover and
		//     no lead-time estimation.
		//   - the demand is non-cyclic: a single stationary demand distribution per
		//     instance, instead of a periodic cycle of distributions.
		// Demand censoring (Kaplan-Meier estimation) is retained but, as in the paper's
		// TED design, activates only in evaluate mode; training always uses the true
		// sampled parameters.
		class MDP
		{
		public:
			double  discount_factor;

			double max_p, min_p, p, h;
			int64_t max_leadtime, min_leadtime, leadtime;
			int64_t MaxOrderSize;
			int64_t MaxSystemInv;

			double mean_demand;
			double stdDemand;
			double max_demand, min_demand;

			bool evaluate;
			bool censoredDemand;

			struct State {
				DynaPlex::StateCategory cat;
				double ServiceLevel;
				int64_t cumulativeStockouts;
				int64_t cumulativeDemands;

				int64_t demand;

				Queue<int64_t> state_vector;
				int64_t total_inv;

				// Deterministic lead time of this instance.
				int64_t leadtime;

				double p;
				int64_t MaxOrderSize, MaxSystemInv, MaxOrderSize_Limit, OrderConstraint;

				// True (sampling) demand distribution of this instance.
				std::vector<double> cumulativePMF;
				int64_t min_true_demand;

				// Kaplan-Meier demand estimator (only used when the MDP's censoredDemand == true).
				// collectDemandStatistics is a dynamic latch: it flips to true the first period
				// on-hand inventory is positive, after which every observation (including censored
				// stockouts) is fed to the estimator. It is genuine trajectory state, not a copy of
				// a config flag, so it cannot be derived from evaluate / censoredDemand.
				bool collectDemandStatistics;
				std::vector<int64_t> past_demands;
				std::vector<int64_t> cumulative_demands;
				std::vector<int64_t> censor_indicator;
				int64_t periodCount;
				std::vector<double> demand_probs;
				int64_t est_min_demand;

				// Estimated demand moments exposed as features (equal to the true moments
				// when demand is not censored).
				double mean_demand;
				double std_demand;

				DynaPlex::VarGroup ToVarGroup() const;
			};

			using Event = int64_t;

			std::vector<double> ReturnUsefulStatistics(const State&) const;
			void ResetHiddenStateVariables(State& state, DynaPlex::RNG&) const;

			//Remainder of the DynaPlex API:
			double ModifyStateWithAction(State&, int64_t action) const;
			double ModifyStateWithEvent(State&, const Event&) const;
			Event GetEvent(const State& state, DynaPlex::RNG&) const;
			DynaPlex::VarGroup GetStaticInfo() const;
			DynaPlex::StateCategory GetStateCategory(const State&) const;
			bool IsAllowedAction(const State&, int64_t action) const;
			State GetInitialState(DynaPlex::RNG& rng) const;
			State GetState(const VarGroup&) const;
			void GetFeatures(const State&, DynaPlex::Features&) const;
			explicit MDP(const DynaPlex::VarGroup&);
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;

		private:
			void UpdateDemandStatistics(State& state, bool uncensored, int64_t newObs) const; // Kaplan - Meier Estimator
			void UpdateOrderLimits(State& state) const;
			// Order-up-to fractiles p/(p+h) of demand-on-leadtime and demand-over-leadtime
			// for a deterministic lead time, given the (estimated or true) demand distribution.
			std::pair<int64_t, int64_t> DemandOverLeadtimeFractiles(const State& state, const DiscreteDist& demand_dist) const;
		};
	}
}
