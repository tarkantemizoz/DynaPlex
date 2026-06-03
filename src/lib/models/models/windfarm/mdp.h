#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"

namespace DynaPlex::Models {
	namespace windfarm /*must be consistent everywhere for complete mdp definition and associated policies and states (if not defined inline).*/
	{
		// A single service engineer maintains windmills spread over several wind farms.
		// Each windmill degrades stochastically through health states
		//   Blue(0) -> Yellow(1) -> Orange(2) -> Red(3),
		// where any non-red state can either degrade one level or jump straight to Red.
		// Red is absorbing until repaired and incurs a penalty every period it persists.
		// Discrete-time, infinite-horizon. Each period the engineer performs exactly one
		// action (travel / service one windmill / idle); afterwards all windmills degrade.
		class MDP
		{
		public:
			// Health-state encoding (also used as indices into the degradation tables).
			static constexpr int64_t Blue = 0;
			static constexpr int64_t Yellow = 1;
			static constexpr int64_t Orange = 2;
			static constexpr int64_t Red = 3;

			double discount_factor;

			int64_t num_farms;            // number of wind farms
			int64_t windmills_per_farm;   // windmills in each (homogeneous) farm
			int64_t num_windmills;        // = num_farms * windmills_per_farm

			// Per-level degradation probabilities. Index 0,1,2 correspond to a windmill
			// currently in Blue, Yellow, Orange respectively.
			std::vector<double> degrade_probs;   // prob. of degrading exactly one level
			std::vector<double> jump_red_probs;  // prob. of jumping straight to Red

			// Costs.
			double travel_cost;        // per travel action
			double maintenance_cost;   // Yellow/Orange -> Blue
			double repair_cost;        // Red -> Blue (typically > maintenance_cost)
			double red_penalty;        // per windmill, per period spent in Red

			// Action layout (totalActions = 1 + num_farms + windmills_per_farm):
			//   0                                  : idle
			//   1 .. num_farms                     : travel to farm (action-1)
			//   num_farms+1 .. num_farms+W         : service local windmill (action-num_farms-1) at current farm
			int64_t totalActions;

			// Exact event model. The unit interval is partitioned into disjoint sub-intervals
			// whose boundaries are the union of every per-state transition threshold. A single
			// uniform draw therefore lands in a sub-interval whose index, combined with the
			// windmill's current health, deterministically yields the next health state. This
			// keeps the event distribution state-independent (required by EventProbabilities).
			std::vector<double> interval_probs;                  // probability mass of each sub-interval
			std::array<std::vector<int64_t>, 4> degradation_map; // degradation_map[health][interval] -> next health

			struct State {
				DynaPlex::StateCategory cat;
				std::vector<int64_t> health;   // health[i] in {Blue,Yellow,Orange,Red}, length num_windmills
				int64_t engineer_location;     // farm index in [0, num_farms)
				DynaPlex::VarGroup ToVarGroup() const;
			};
			// One sub-interval index per windmill.
			using Event = std::vector<int64_t>;

			double ModifyStateWithAction(State&, int64_t action) const;
			double ModifyStateWithEvent(State&, const Event&) const;
			Event GetEvent(DynaPlex::RNG& rng) const;
			std::vector<std::tuple<Event, double>> EventProbabilities() const;
			DynaPlex::VarGroup GetStaticInfo() const;
			DynaPlex::StateCategory GetStateCategory(const State&) const;
			bool IsAllowedAction(const State& state, int64_t action) const;
			State GetInitialState() const;
			State GetState(const VarGroup&) const;
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;
			void GetFeatures(const State&, DynaPlex::Features&) const;
			explicit MDP(const DynaPlex::VarGroup&);

			// Helper: the farm a global windmill index belongs to.
			int64_t FarmOf(int64_t windmill) const { return windmill / windmills_per_farm; }
		};
	}
}
