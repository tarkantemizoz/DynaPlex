#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"

namespace DynaPlex::Models {
	namespace random_leadtimes 
	{
		class MDP
		{
		public:
			double  discount_factor;

			double DemandRate;

			double b, h;
			double ExponentialLeadTimeRate;
			double UniformStart;
			double UniformEnd;
			double q;
			double tau;

			int64_t MaxOrderSize;
			int64_t BaseStockLevel;
			int64_t MaxOrdersInPipeline;
			int64_t MaxAllowedBackorder;
			int64_t MaxInventoryPosition;
			int64_t MinInventoryPosition;
			int64_t InitialInventoryLevel;
			int64_t InitialOrdersInPipeline;
			
			std::string lead_time_dist;
			bool exponential_lead_time_dist, optimized;

			double Kmax;
			double Kmin;
			double Kbo;

			struct State {
				DynaPlex::StateCategory cat;

				int64_t InventoryLevel;
				int64_t OrdersInPipeline;
				int64_t InventoryPosition;
				std::vector<double> OrderTimes;

				DynaPlex::VarGroup ToVarGroup() const;
			};
  
			using Event = std::pair<double, std::vector<double>>;

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
		};
	}
}