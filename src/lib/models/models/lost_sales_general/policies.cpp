#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
#include "dynaplex/trajectory.h"


namespace DynaPlex::Models {
	namespace lost_sales_general
	{
		GreedyCappedBaseStockPolicy::GreedyCappedBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
		}

		int64_t GreedyCappedBaseStockPolicy::GetAction(const MDP::State& state) const
		{
			return state.OrderConstraint;
		}
	}
}