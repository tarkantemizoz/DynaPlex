#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
namespace DynaPlex::Models {
	namespace random_leadtimes 
	{
		BaseStockPolicy::BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.Get("base_stock_level", base_stock_level);
		}

		int64_t BaseStockPolicy::GetAction(const MDP::State& state) const
		{
			return std::max((int64_t)0, base_stock_level - state.InventoryPosition);
		}

		InitialPolicy::InitialPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
		}

		int64_t InitialPolicy::GetAction(const MDP::State& state) const
		{
			int64_t action = std::max((int64_t)0, mdp->BaseStockLevel - state.InventoryPosition);
			if (action > 0)
			{
				if (action > mdp->MaxOrderSize)
					action = mdp->MaxOrderSize;

				while (action + state.OrdersInPipeline > mdp->MaxOrdersInPipeline && action > 0)
				{
					action--;
				}
			}

			return action;
		}
	}
}