#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"

namespace DynaPlex::Models {
	namespace lost_sales_all
	{
		class MDP
		{
		public:
			double  discount_factor;

			double max_p, min_p, h, p;
			int64_t max_leadtime, min_leadtime;
			int64_t max_num_cycles;
			int64_t MaxOrderSize;
			int64_t MaxSystemInv;

			std::vector<double> mean_demand;
			std::vector<double> stdDemand;
			std::vector<double> leadtime_probs;
			double max_demand, min_demand;

			bool evaluate;
			bool returnRewards;
			bool collectStatistics;
			bool censoredDemand;
			bool censoredLeadtime;

			std::vector<double> fractiles;
			std::vector<int64_t> demand_cycles;

			struct State {
				DynaPlex::StateCategory cat;

				int64_t period;
				std::vector<int64_t> demand_cycles;
				int64_t limitingPeriod;
				bool demand_waits_order;
				int64_t order_initializationPhase;

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

				bool collectStatistics, censoredDemand, stochasticLeadtimes, censoredLeadtime;

				std::vector<std::vector<double>> true_demand_probs;
				std::vector<int64_t> max_true_demand;
				std::vector<int64_t> min_true_demand;

				double p;
				int64_t MaxOrderSize, MaxSystemInv, MaxOrderSize_Limit;
				std::vector<int64_t> fractiles;
				std::vector<std::vector<int64_t>> cycle_fractiles;
				std::vector<int64_t> cycle_MaxOrderSize;
				std::vector<int64_t> cycle_MaxSystemInv;

				std::vector<double> leadtime_probs;
				std::vector<double> estimated_leadtime_probs;
				std::vector<double> cumulative_leadtime_probs;
				int64_t min_leadtime, max_leadtime;
				int64_t estimated_min_leadtime, estimated_max_leadtime;
				std::vector<int64_t> past_leadtimes;
				int64_t orders_received;

				DynaPlex::VarGroup ToVarGroup() const;
			};
  
			using Event = std::pair<int64_t, std::vector<double>>;

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
			//Enables all MDPs to be constructer in a uniform manner.
			explicit MDP(const DynaPlex::VarGroup&);
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;

			int64_t GetL(const State&) const;
			int64_t GetReinitiateCounter(const State&) const;
			int64_t GetH(const State&) const;
			int64_t GetM(const State&) const;

		private:
			void UpdateDemandStatistics(State& state, bool uncensored, int64_t newObs, int64_t current_cyclePeriod) const;
			void UpdateLeadTimeStatistics(State& state) const;
			void UpdateOrderLimits(State& state) const;

			std::vector<double> SampleLeadTimeDistribution(RNG& rng, int64_t min_lt, int64_t max_lt) const;
		};
	}
}