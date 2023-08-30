#include "lostsalesmdp.h"
#include "dynaplex/mdpregistrar.h"

namespace DynaPlex::Models {
	using namespace DynaPlex;
	namespace LostSales /*keep this in line with id below and with namespace name in header*/
	{
		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;		
			vars.Add("num_valid_actions", MaxOrderSize + 1);
			vars.Add("num_features", leadtime);

			//potentially add any stuff that was computed for diagnostics purposes
			//not used by dynaplex framework itself. 
			VarGroup diagnostics{};			
			diagnostics.Add("MaxOrderSize", MaxOrderSize);
			diagnostics.Add("MaxSystemInv", MaxSystemInv);
			vars.Add("diagnostics", diagnostics);
			
			return vars;
		}

		double MDP::ModifyStateWithAction(State& state, int64_t action) const
		{
			state.state_vector.push_back(action);
			state.total_inv += action;
			state.state_type = StateType::AwaitEvent;
			return 0.0;
		}


		MDP::State MDP::GetInitialState() const
		{
			auto queue = Queue<int64_t>{};
			queue.reserve(leadtime + 1);
			queue.push_back(MaxSystemInv);//<- initial on-hand
			for (size_t i = 0; i < leadtime - 1; i++)
			{
				queue.push_back(0);
			}
			return State{ StateType::AwaitEvent,queue,queue.sum() };
		}

		MDP::MDP(const VarGroup& varGroup) :
			varGroup{ varGroup }
		{
			varGroup.Get("p", p);
			varGroup.Get("h", h);
			varGroup.Get("leadtime", leadtime);
			varGroup.Get("demand_dist",demand_dist);


			//Initiate members that are computed from the parameters:
			auto DemOverLeadtime = DiscreteDist::GetZeroDist();
			for (size_t i = 0; i <= leadtime; i++)
			{
				DemOverLeadtime = DemOverLeadtime.Add(demand_dist);
			}
			MaxOrderSize = demand_dist.Fractile(p / (p + h));
			MaxSystemInv = DemOverLeadtime.Fractile(p / (p + h));
		}


		MDP::Event MDP::GetEvent(RNG& rng) const {
			return demand_dist.GetSample(rng);
		}

		double MDP::ModifyStateWithEvent(State& state, const MDP::Event& event) const
		{
			state.state_type= StateType::AwaitAction;

			auto onHand = state.state_vector.pop_front();//Length is L again.

			if (onHand > event)
			{//There is sufficient inventory. Satisfy order and incur holding costs
				onHand -= event;
				state.total_inv -= event;
				state.state_vector.front() += onHand;
				return onHand * h;
			}
			else
			{
				state.total_inv -= onHand;
				return (event - onHand) * p;
			}
		}

		DynaPlex::StateType MDP::GetStateType(const State& state) const
		{
			return state.state_type;
		}
		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
			if (state.total_inv + action <= MaxSystemInv)
			{
				return true;
			}
			return false;
		}
	}

	//static registrar (only in .cpp file): includes MDP in the central registry, such that DynaPlex::GetMDP can locate it 
	static MDPRegistrar<LostSales::MDP> registrar(/*id*/"lost_sales",/*optional brief description*/ "Canonical lost sales problem.");
}

