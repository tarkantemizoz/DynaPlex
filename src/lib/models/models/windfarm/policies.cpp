#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"

namespace DynaPlex::Models {
	namespace windfarm /*keep this namespace name in line with the namespace in which the mdp corresponding to this policy is defined*/
	{
		GreedyEngineerPolicy::GreedyEngineerPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			: mdp{ mdp }
		{
		}

		int64_t GreedyEngineerPolicy::GetAction(const MDP::State& state) const
		{
			const int64_t W = mdp->windmills_per_farm;
			const int64_t F = mdp->num_farms;
			const int64_t loc = state.engineer_location;

			// 1) Most urgent windmill at the current farm (higher health index = more urgent).
			int64_t best_local = -1;
			int64_t best_local_health = MDP::Blue;
			for (int64_t j = 0; j < W; ++j)
			{
				const int64_t h = state.health[loc * W + j];
				if (h > best_local_health)
				{
					best_local_health = h;
					best_local = j;
				}
			}
			if (best_local >= 0) // something here needs servicing
				return 1 + F + best_local; // service local windmill best_local

			// 2) Nothing to do here: find the farm with the most urgent windmill overall.
			int64_t best_farm = -1;
			int64_t best_farm_health = MDP::Blue;
			for (int64_t f = 0; f < F; ++f)
			{
				if (f == loc)
					continue;
				for (int64_t j = 0; j < W; ++j)
				{
					const int64_t h = state.health[f * W + j];
					if (h > best_farm_health)
					{
						best_farm_health = h;
						best_farm = f;
					}
				}
			}
			if (best_farm >= 0)
				return 1 + best_farm; // travel to that farm

			// 3) Everything is healthy: idle.
			return 0;
		}
	}
}
