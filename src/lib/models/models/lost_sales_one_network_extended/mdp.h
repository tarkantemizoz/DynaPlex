#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"

namespace DynaPlex::Models {
	namespace lost_sales_one_network_extended 
	{		
		class MDP
		{
		public:			
			double  discount_factor;

			double max_p, min_p, h, p;
			int64_t max_leadtime, min_leadtime, leadtime;
			int64_t MaxOrderSize, MaxSystemInv;
			bool evaluate;
			double mean_demand, stdev_demand;
			double max_demand, min_demand;

			struct State {
				DynaPlex::StateCategory cat;

				//Other members depend on the MDP:
				Queue<int64_t> state_vector;
				int64_t total_inv;
				double p;
				int64_t leadtime, MaxOrderSize, MaxSystemInv;
				double mean_demand, stdev_demand;
				std::vector<double> demand_probs{};
				int64_t max_demand, min_demand;

				//declaration; for definition see mdp.cpp:
				DynaPlex::VarGroup ToVarGroup() const;
			};

  
			using Event = int64_t;
			double ModifyStateWithAction(State&, int64_t action) const;
			double ModifyStateWithEvent(State&,const Event&) const;
			Event GetEvent(const State& state, DynaPlex::RNG&) const;
			DynaPlex::VarGroup GetStaticInfo() const;			
			DynaPlex::StateCategory GetStateCategory(const State&) const;			
			bool IsAllowedAction(const State&, int64_t action) const;			
			State GetInitialState(DynaPlex::RNG&) const;
			State GetState(const VarGroup&) const;		
			void GetFeatures(const State&, DynaPlex::Features&) const;			
			//Enables all MDPs to be constructer in a uniform manner.
			explicit MDP(const DynaPlex::VarGroup&);
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;

			int64_t GetL(const State&) const;
			int64_t GetReinitiateCounter(const State&) const;
			int64_t GetH(const State&) const;
			int64_t GetM(const State&) const;
		};
	}
}

