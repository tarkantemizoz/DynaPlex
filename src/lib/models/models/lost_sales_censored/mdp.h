#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"

namespace DynaPlex::Models {
	namespace lost_sales_censored
	{
		class MDP
		{
		public:
			double  discount_factor;

			double p, h;
			int64_t leadtime;
			int64_t MaxOrderSize;
			int64_t MaxSystemInv, MinSystemInv;
			int64_t statTreshold;
			int64_t updateCycle;

			double mean_demand, stdDemand;
			double max_demand, min_demand;

			bool evaluate;
			bool returnRewards;
			bool collectStatistics;

			std::vector<std::vector<int64_t>> ordervector;
			std::vector<std::vector<int64_t>> invvector;
			std::vector<double> fractiles;
			std::vector<double> demand_vector;
			std::vector<std::vector<double>> std_vector;

			struct State {
				DynaPlex::StateCategory cat;

				Queue<int64_t> state_vector;
				int64_t total_inv;

				std::vector<int64_t> past_demands;
				std::vector<int64_t> cumulative_demands;
				std::vector<int64_t> censor_indicator;
				std::vector<int64_t> fractiles;

				int64_t periodCount;
				int64_t uncensoredperiodCount;
				int64_t initialperiodCount;
				int64_t statTreshold;
				int64_t updateCycle;
				double meanDemand;
				double stdDemand;
				double meanUncensoredDemand;
				double stdUncensoredDemand;
				bool collectStatistics;

				std::vector<double> true_demand_probs;
				int64_t max_true_demand, min_true_demand;

				int64_t MaxOrderSize, MaxSystemInv, MinSystemInv;
				DynaPlex::VarGroup ToVarGroup() const;
			};

  
			using Event = int64_t;

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

			int64_t findGreaterIndex(const std::vector<double>& v, double x) const;
			void UpdateStatistics(State& state, bool uncensored, int64_t newObs) const;

			int64_t GetReinitiateCounter(const State& state) const;
		};
	}
}