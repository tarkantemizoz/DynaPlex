#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"

namespace DynaPlex::Models {
	namespace lost_sales_general
	{
		class MDP
		{
		public:
			double  discount_factor;

			double p, h;
			bool stochasticLeadtimes, order_crossover;
			int64_t max_leadtime, min_leadtime;
			std::vector<double> cumulative_leadtime_probs;
			
			int64_t MaxOrderSize;
			std::vector<int64_t> cycle_MaxOrderSize;
			std::vector<int64_t> cycle_MaxSystemInv;
			int64_t cycle_length;

			std::vector<std::vector<double>> cumulativePMFs;
			std::vector<int64_t> min_true_demand;

			bool maximizeRewards;

			struct State {
				DynaPlex::StateCategory cat;
				double ServiceLevel;
				int64_t cumulativeStockouts;
				int64_t cumulativeDemands;

				int64_t period;
				Queue<int64_t> state_vector;
				int64_t total_inv;
				int64_t OrderConstraint;

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
			State GetInitialState() const;
			State GetState(const VarGroup&) const;
			void GetFeatures(const State&, DynaPlex::Features&) const;
			explicit MDP(const DynaPlex::VarGroup&);
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;

			int64_t GetL(const State&) const;
			int64_t GetReinitiateCounter(const State&) const;
			int64_t GetH(const State&) const;
			int64_t GetM(const State&) const;
		};
	}
}