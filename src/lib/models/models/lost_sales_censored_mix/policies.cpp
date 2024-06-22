#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
namespace DynaPlex::Models {
	namespace lost_sales_censored_mix
	{
		ConstantOrderPolicy::ConstantOrderPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.Get("co_level", co_level);
		}

		int64_t ConstantOrderPolicy::GetAction(const MDP::State& state) const
		{
			return co_level;
		}

		BaseStockPolicy::BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.Get("base_stock_level", base_stock_level);
		}

		int64_t BaseStockPolicy::GetAction(const MDP::State& state) const
		{
			int64_t action = base_stock_level - state.total_inv;
			if (action < 0)
			{
				action = 0;
			}
			return action;
		}

		CappedBaseStockPolicy::CappedBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.Get("S", S);
			config.Get("r", r);
		}

		int64_t CappedBaseStockPolicy::GetAction(const MDP::State& state) const
		{
			int64_t action = S - state.total_inv;
			if (action > r)
			{
				action = r;
			}
			if (action < 0)
			{
				action = 0;
			}
			return action;
		}

		GreedyCappedBaseStockPolicy::GreedyCappedBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
		}

		int64_t GreedyCappedBaseStockPolicy::GetAction(const MDP::State& state) const
		{
			int64_t action = state.MaxSystemInv - state.total_inv;
			if (action > state.MaxOrderSize)
			{
				action = state.MaxOrderSize;
			}
			if (action < 0)
			{
				action = 0;
			}
			return action;
		}
	}
}