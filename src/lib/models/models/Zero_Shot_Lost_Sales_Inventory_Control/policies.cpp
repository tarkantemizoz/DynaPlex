#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
#include "dynaplex/trajectory.h"


namespace DynaPlex::Models {
	namespace Zero_Shot_Lost_Sales_Inventory_Control
	{
		//DynamicBaseStockPolicies::DynamicBaseStockPolicies(std::shared_ptr<const MDP> mdp, const VarGroup& config)
		//	:mdp{ mdp }
		//{
		//	config.Get("co_level", co_level);
		//}

		//int64_t DynamicBaseStockPolicies::GetAction(const MDP::State& state) const
		//{



		//	using dp_State = std::unique_ptr<StateBase>;
		//	dp_State dummystate = state;
		//	auto& t_state = ToState(state);
		//	dp_State other = state->Clone()


		//	std::vector<DynaPlex::Trajectory> trajectories{};
		//	trajectories.reserve(100);


		//	for (int64_t experiment_number = 0; experiment_number < 100; experiment_number++)
		//	{
		//		trajectories.emplace_back(experiment_number + 0);
		//		trajectories.back().RNGProvider.SeedEventStreams(true, 123, experiment_number + 213);
		//	}
		//	for (DynaPlex::Trajectory& traj : trajectories)
		//	{
		//		traj.Reset(state);
		//		auto& t_state = ToState(traj.GetState());
		//		traj.Category = mdp->GetStateCategory(t_state);
		//	}

		//	return co_level;
		//}


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
			return state.OrderConstraint;
		}
	}
}