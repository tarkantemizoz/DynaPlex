#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"

namespace DynaPlex::Models {
	namespace Zero_Shot_Lost_Sales_Inventory_Control
	{
		class MDP
		{
		public:
			double  discount_factor;

			double max_p, min_p, p, h;
			int64_t max_leadtime, min_leadtime;
			int64_t max_num_cycles;
			int64_t MaxOrderSize;
			int64_t MaxSystemInv;

			std::vector<int64_t> demand_cycles;
			std::vector<double> mean_demand;
			std::vector<double> stdDemand;
			std::vector<double> leadtime_probs;
			std::vector<double> non_crossing_leadtime_rv_probs;
			double max_demand, min_demand, max_period_demand;

			bool evaluate;
			bool maximizeRewards;
			bool train_stochastic_leadtimes;
			bool train_cyclic_demand;
			bool censoredDemand;
			bool censoredLeadtime;
			bool order_crossover;

			struct State {
				DynaPlex::StateCategory cat;
				double ServiceLevel;
				int64_t cumulativeStockouts;
				int64_t cumulativeDemands;

				int64_t period;
				int64_t demand;
				int64_t cycle_length;
				std::vector<int64_t> demand_cycles;
				std::vector<bool> collectDemandStatistics;

				Queue<int64_t> state_vector;
				int64_t total_inv;

				std::vector<std::vector<int64_t>> past_demands;
				std::vector<std::vector<int64_t>> cumulative_demands;
				std::vector<std::vector<int64_t>> censor_indicator;
				std::vector<std::vector<double>> cycle_probs;
				std::vector<int64_t> cycle_min_demand;
				std::vector<double> mean_cycle_demand;
				std::vector<double> std_cycle_demand;
				std::vector<int64_t> periodCount;

				bool collectStatistics, censoredDemand;
				bool stochasticLeadtimes, censoredLeadtime, order_crossover;

				std::vector<std::vector<double>> cumulativePMFs;
				std::vector<int64_t> min_true_demand;

				double p;
				int64_t MaxOrderSize, MaxSystemInv, MaxOrderSize_Limit, OrderConstraint;
				std::vector<int64_t> cycle_MaxOrderSize;
				std::vector<int64_t> cycle_MaxSystemInv;

				std::vector<double> estimated_leadtime_probs;
				std::vector<double> cumulative_leadtime_probs;
				int64_t min_leadtime, max_leadtime;
				int64_t estimated_min_leadtime, estimated_max_leadtime;
				std::vector<int64_t> past_leadtimes;
				int64_t orders_received;

				DynaPlex::VarGroup ToVarGroup() const;
			};
  
			using Event = std::pair<int64_t, std::vector<double>>;

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
			void UpdateLeadTimeStatistics(State& state) const;
			void UpdateOrderLimits(State& state) const;
			// Order-up-to fractiles p/(p+h) of demand-on-leadtime and demand-over-leadtime, for a
			// trajectory starting at base_period, given the per-cycle-period demand distributions.
			// Shared by GetInitialState (per cycle period) and UpdateOrderLimits (current period).
			std::pair<int64_t, int64_t> DemandOverLeadtimeFractiles(const State& state, int64_t base_period, const std::vector<DiscreteDist>& cycle_demand_dists, const std::vector<double>& leadtime_probs) const;

			std::vector<double> SampleLeadTimeDistribution(RNG& rng, int64_t min_lt, int64_t max_lt) const;
		};
	}
}