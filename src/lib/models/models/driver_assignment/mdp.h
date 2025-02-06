#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"

namespace DynaPlex::Models {
	namespace driver_assignment
	{
		class MDP
		{
		public:
			int64_t number_drivers;
			double mean_order_per_time_unit; // interarrival time b/w order arrivals

			// Defining the grid
			double max_x;
			double min_x = 0;
			double max_y;
			double min_y = 0;

			double max_working_hours; // then overtime
			double horizon_length; // this is to end the mdp, which is defined infinite and triggered by events (order arrivals)


			double cost_per_km;
			double penalty_for_workhour_violation; // per distance(km) in overtime
			//double timewindow_duration;

			struct State {
				//using this is recommended:
				DynaPlex::StateCategory cat;
				//driver state attributes: (is_busy, loc_x, loc_y, next_time_avail, remaining_work_hours)
				std::vector<double> current_driver_list_conc;
				//order state attributes: (pickup_loc_x, pickup_loc_y, deliver_loc_x, deliver_loc_y)
				std::vector<double> current_order_typelist;

				double current_time;
				DynaPlex::VarGroup ToVarGroup() const;
			};

			using Event = std::vector<double>;

			double ModifyStateWithAction(State&, int64_t action) const;
			double ModifyStateWithEvent(State&, const Event&) const;
			Event GetEvent(DynaPlex::RNG& rng) const;
			//std::vector<std::tuple<Event,double>> EventProbabilities() const; // needed if we use exact solver
			DynaPlex::VarGroup GetStaticInfo() const;
			DynaPlex::StateCategory GetStateCategory(const State&) const;
			bool IsAllowedAction(const State& state, int64_t action) const;
			State GetInitialState(DynaPlex::RNG& rng) const;
			State GetState(const VarGroup&) const;
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;
			void GetFeatures(const State&, DynaPlex::Features&) const;
			explicit MDP(const DynaPlex::VarGroup&);
		};
	}
}