#pragma once
#include "memory"
#include "dynaplex/rng.h"
#include "dynaplex/vargroup.h"
#include "erasure_concepts.h"
#include "actionrangeprovider.h"


namespace DynaPlex::Erasure
{
	template <class t_MDP, double objective>
	class GreedyPolicy
	{
		static_assert(HasGetStaticInfo<t_MDP>, "MDP must publicly define GetStaticInfo() const returning DynaPlex::VarGroup.");

		DynaPlex::Erasure::ActionRangeProvider<t_MDP> provider;
		std::shared_ptr<const t_MDP> mdp;
	public:

		GreedyPolicy(std::shared_ptr<const t_MDP> mdp, const DynaPlex::VarGroup& varGroup)
			: provider{ mdp }, mdp{ mdp }
		{
		}
		using State = t_MDP::State;

		int64_t GetAction(const State& state, DynaPlex::RNG& rng) const
		{
			if constexpr (DynaPlex::Erasure::HasModifyStateWithAction<t_MDP>)
			{
				double best_direct_return = -1.0 * std::numeric_limits<double>::infinity();
				int64_t best_action = -1;
				bool action_found = false;
				for (auto action : provider(state))
				{
					State copy{ state };
					double direct_normalized_return = mdp->ModifyStateWithAction(copy, action) * objective;
					if (direct_normalized_return > best_direct_return)
					{
						best_direct_return = direct_normalized_return;
						best_action = action;
						action_found = true;
					}
				}
				if (action_found)
					return best_action;
				else
					throw DynaPlex::Error("GreedyPolicy: No action found; maybe state does not allow a single action?");
			}
			else
				throw DynaPlex::Error("GreedyPolicy: \nMDP does not publicly define ModifyStateWithAction(MDP::State,int64_t) const returning double");

		}
	};
}