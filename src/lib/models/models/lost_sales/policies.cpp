#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
namespace DynaPlex::Models {
	namespace lost_sales /*keep this namespace name in line with the name space in which the mdp corresponding to this policy is defined*/
	{

		//MDP and State refer to the specific ones defined in current namespace
		BaseStockPolicy::BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.GetOrDefault("base_stock_level", base_stock_level, mdp->MaxSystemInv);
			config.GetOrDefault("capped", capped, true);
		}

		int64_t BaseStockPolicy::GetAction(const MDP::State& state) const
		{
			if (base_stock_level > state.total_inv) {
				int64_t action = base_stock_level - state.total_inv;
				//We maximize, so actually this is capped base-stock. 
				if (action > mdp->MaxOrderSize && capped)
				{
					action = mdp->MaxOrderSize;
				}
				return action;
			}
			else {
				return 0;
			}
		}
	}
}