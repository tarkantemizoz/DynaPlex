#pragma once
#include <cstdint>
#include "mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Models {
	namespace windfarm /*must be consistent everywhere for complete mdp definition and associated policies.*/
	{
		// Greedy benchmark: prioritise the most-degraded windmill reachable right now.
		// - If the current farm has a windmill needing work, service the most urgent one
		//   (Red before Orange before Yellow).
		// - Otherwise travel towards the farm holding the most urgent windmill.
		// - If every windmill is healthy, idle.
		class GreedyEngineerPolicy
		{
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
		public:
			GreedyEngineerPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};
	}
}
