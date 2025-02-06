#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
namespace DynaPlex::Models {
	namespace multi_item_sla
	{
		BaseStockPolicy::BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.GetOrDefault("serviceLevelPolicy", serviceLevelPolicy, mdp->benchmarkAction);
		}

		int64_t BaseStockPolicy::GetAction(const MDP::State& state) const
		{
			int64_t action = serviceLevelPolicy + 1;
			return action;
		}

		DynamicPolicy::DynamicPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.Get("bestPolicyFillRate", bestPolicyFillRate);
			config.Get("policyFillRates", policyFillRates);
		}

		int64_t DynamicPolicy::GetAction(const MDP::State& state) const
		{
			int64_t satisfiedDemands = state.ObservedDemand - state.CumulativeStockouts;
			int64_t totalExpectedDemands = state.TimeRemaining * mdp->totalDemandRate;
			int64_t totalShouldSatisfy = static_cast<int64_t>(std::ceil(bestPolicyFillRate * (totalExpectedDemands + state.ObservedDemand))) - satisfiedDemands;
			int64_t totalActions = mdp->totalActions - 1;

			for (int64_t i = 0; i < totalActions; i++) {
				int64_t expectedSatisfy = static_cast<int64_t>(std::floor(policyFillRates[i] * totalExpectedDemands));
				if (expectedSatisfy >= totalShouldSatisfy) {
					return i + 1;
				}
			}
			return totalActions;
		}
	}
}