#include "lostsalesmdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "lostsalespolicies.h"

namespace DynaPlex::Models {
	using namespace DynaPlex;
	namespace lost_sales /*keep this in line with id below and with namespace name in header*/
	{
		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;		
			vars.Add("valid_actions", MaxOrderSize + 1);
			
			VarGroup feats{};
			
			vars.Add("features", feats);

			vars.Add("discount_factor", discount_factor);

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
			state.cat = StateCategory::AwaitEvent();
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
			State state{};
			state.cat = StateCategory::AwaitEvent();
			state.state_vector = queue;
			state.total_inv = queue.sum();
			return state;
		}

		MDP::MDP(const VarGroup& varGroup) :
			varGroup{ varGroup }
		{
			varGroup.Get("p", p);
			varGroup.Get("h", h);
			varGroup.Get("leadtime", leadtime);
			varGroup.Get("demand_dist",demand_dist);
			
			//providing discount_factor is optional. 
			if (varGroup.HasKey("discount_factor"))
				varGroup.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;
			

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
			state.cat= StateCategory::AwaitAction();

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

		void MDP::GetFeatures(State&, DynaPlex::Features&) {
			throw DynaPlex::Error("get features not implemented on lost_sales");
		}
		

		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{//Here, we register any custom heuristics we want to provide for this MDP.	
		 //On the generic DynaPlex::MDP constructed from this, these heuristics can be obtained
		 //in generic form using mdp->GetPolicy(VarGroup vars), with the id in var set
		 //to the corresponding id given below.
			registry.Register<BaseStockPolicy>("basestock",
				"Base-stock policy with fixed, non-adjustable base-stock level equal"
				" to the bound on system inventory discussed in Zipkin (2008)");
		}

	

		void MDP::GetFeatures(const State&, Features&) const
		{
			return;
		}


		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
			if (state.total_inv + action <= MaxSystemInv)
			{
				return true;
			}
			return false;
		}


		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "lost_sales",
				/*description*/ "Canonical lost sales problem, see e.g. Zipkin (2008) for a formal description. (parameters: p, h, leadtime, demand_dist.)",
				registry); 
		}
	}
}
