#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"

namespace DynaPlex::Models {
	namespace lost_sales_cyclic
	{
		class MDP
		{
		public:
			double  discount_factor;

			double max_p, min_p, h, p;
			int64_t max_leadtime, min_leadtime, leadtime;
			int64_t max_num_cycles;
			int64_t MaxOrderSize;
			int64_t MaxSystemInv;

			std::vector<double> mean_demand;
			std::vector<double> stdDemand;
			double max_demand, min_demand;

			bool evaluate;
			bool returnRewards;
			bool collectStatistics;
			bool censoredDemand;

			std::vector<double> fractiles;
			std::vector<int64_t> demand_cycles;

			struct State {
				DynaPlex::StateCategory cat;

				int64_t period;
				std::vector<int64_t> demand_cycles;
				int64_t limitingPeriod;
				int64_t order_initializationPhase;

				Queue<int64_t> state_vector;
				int64_t total_inv;

				std::vector<std::vector<int64_t>> past_demands;
				std::vector<std::vector<int64_t>> cumulative_demands;
				std::vector<std::vector<int64_t>> censor_indicator;

				std::vector<std::vector<double>> cycle_probs;
				std::vector<double> mean_cycle_demand;
				std::vector<double> std_cycle_demand;

				std::vector<int64_t> periodCount;
				bool collectStatistics, censoredDemand;

				std::vector<std::vector<double>> true_demand_probs;
				std::vector<int64_t> max_true_demand;
				std::vector<int64_t> min_true_demand;

				double p;
				int64_t leadtime, MaxOrderSize, MaxSystemInv;
				std::vector<int64_t> fractiles;
				std::vector<std::vector<int64_t>> cycle_fractiles;
				std::vector<int64_t> cycle_MaxOrderSize;
				std::vector<int64_t> cycle_MaxSystemInv;

				DynaPlex::VarGroup ToVarGroup() const;

				void AddNestedVector(DynaPlex::VarGroup& vg, const std::string& key, const std::vector<std::vector<int64_t>>& nestedVec) const;
				void AddNestedVectorDouble(DynaPlex::VarGroup& vg, const std::string& key, const std::vector<std::vector<double>>& nestedVec) const;
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

			int64_t GetL(const State&) const;
			int64_t GetReinitiateCounter(const State&) const;
			int64_t GetH(const State&) const;
			int64_t GetM(const State&) const;

			void UpdateStatistics(State& state, bool uncensored, int64_t newObs) const;

			//void AddNestedVector(DynaPlex::VarGroup& vg, const std::string& key, const std::vector<std::vector<int64_t>>& nestedVec) const;
			void GetNestedVector(const DynaPlex::VarGroup& vg, const std::string& key, std::vector<std::vector<int64_t>>& outNestedVec) const;
			//void AddNestedVectorDouble(DynaPlex::VarGroup& vg, const std::string& key, const std::vector<std::vector<double>>& nestedVec) const;
			void GetNestedVectorDouble(const DynaPlex::VarGroup& vg, const std::string& key, std::vector<std::vector<double>>& outNestedVec) const;
		};
	}
}