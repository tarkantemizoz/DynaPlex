#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace lost_sales_one_network_extended /*keep this in line with id below and with namespace name in header*/
	{
		int64_t MDP::GetL(const State& state) const
		{
			return static_cast<int64_t>(state.leadtime * 10);
		}

		int64_t MDP::GetReinitiateCounter(const State& state) const
		{
			return static_cast<int64_t>(state.leadtime * 50);
		}

		int64_t MDP::GetH(const State& state) const
		{
			return static_cast<int64_t>(state.leadtime * 5);
		}

		int64_t MDP::GetM(const State& state) const
		{
			return static_cast<int64_t>(state.leadtime * state.mean_demand * 20);
		}

		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;		
			vars.Add("valid_actions", MaxOrderSize + 1);			
			vars.Add("discount_factor", discount_factor);

			VarGroup diagnostics{};			
			diagnostics.Add("MaxOrderSize", MaxOrderSize);
			diagnostics.Add("MaxSystemInv", MaxSystemInv);
			vars.Add("diagnostics", diagnostics);
			
			return vars;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
			return ((state.total_inv + action) <= state.MaxSystemInv && action <= state.MaxOrderSize)
				|| action == 0;
		}

		double MDP::ModifyStateWithAction(State& state, int64_t action) const
		{
			state.state_vector.push_back(action);
			state.total_inv += action;
			state.cat = StateCategory::AwaitEvent();
			return 0.0;
		}

		MDP::MDP(const VarGroup& config)
		{
			config.Get("max_p", max_p);
			config.Get("max_leadtime", max_leadtime);
			config.Get("evaluate", evaluate);
			config.Get("max_demand", max_demand);
			config.Get("min_demand", min_demand);

			h = 1.0;
			min_p = 2.0;
			min_leadtime = 2;

			if (config.HasKey("leadtime"))
				config.Get("leadtime", leadtime);
			else
				leadtime = min_leadtime;

			if (config.HasKey("p"))
				config.Get("p", p);
			else
				p = min_p;

			if (config.HasKey("mean_demand"))
				config.Get("mean_demand", mean_demand);
			else
				mean_demand = max_demand;

			if (config.HasKey("stdev_demand"))
				config.Get("stdev_demand", stdev_demand);
			else
				stdev_demand = mean_demand;

			//providing discount_factor is optional. 
			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;			

			DynaPlex::DiscreteDist dist = DiscreteDist::GetAdanEenigeResingDist(max_demand, max_demand * 2);
			//Initiate members that are computed from the parameters:
			auto DemOverLeadtime = DiscreteDist::GetZeroDist();
			for (size_t i = 0; i <= max_leadtime; i++)
			{
				DemOverLeadtime = DemOverLeadtime.Add(dist);
			}
			MaxOrderSize = dist.Fractile(max_p / (max_p + h));
			MaxSystemInv = DemOverLeadtime.Fractile(max_p / (max_p + h));
		}

		MDP::Event MDP::GetEvent(const State& state, RNG& rng) const {
			// Generate a uniform random number between 0 and 1
			double randomValue = rng.genUniform();
			double cumulativeProbability = 0.0;
			for (size_t i = 0; i < state.demand_probs.size(); i++) {
				cumulativeProbability += state.demand_probs[i];
				if (randomValue < cumulativeProbability) {
					return state.min_demand + static_cast<int64_t>(i);
				}
			}
			return state.max_demand;
		}

		double MDP::ModifyStateWithEvent(State& state,const MDP::Event& event) const
		{
			state.cat= StateCategory::AwaitAction();

			auto onHand = state.state_vector.pop_front();//Length is state.leadtime again.
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
				return (event - onHand) * state.p;
			}
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			features.Add(state.state_vector);
			for (int64_t i = 0; i < max_leadtime - state.leadtime; i++) {
				features.Add(0);
			}
			features.Add(state.leadtime);
			features.Add(state.p);
			features.Add(state.mean_demand);
			features.Add(state.stdev_demand);
		}
		
		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{
			registry.Register<ConstantOrderPolicy>("constant_order",
				"Constant order policy with parameter co_level.");
			registry.Register<BaseStockPolicy>("base_stock",
				"Base-stock policy with parameter base_stock_level.");
			registry.Register<CappedBaseStockPolicy>("capped_base_stock",
				"Capped base-stock policy with parameters S and r.");
			registry.Register<GreedyCappedBaseStockPolicy>("greedy_capped_base_stock",
				"Capped base-stock policy with suboptimal S and r.");
		}
		
		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		MDP::State MDP::GetInitialState(RNG& rng) const
		{
			State state{};
			if (evaluate) {
				state.p = p;
				state.leadtime = leadtime;
				state.mean_demand = mean_demand;
				state.stdev_demand = stdev_demand;
			}
			else {
				double randomValue_p = rng.genUniform();
				state.p = randomValue_p * (max_p - min_p) + min_p;
				double randomValue_leadtime = rng.genUniform();
				state.leadtime = static_cast<int64_t>(std::floor(randomValue_leadtime * (max_leadtime - min_leadtime + 1))) + min_leadtime;
				double randomValue_mean = rng.genUniform();
				state.mean_demand = randomValue_mean * (max_demand - min_demand) + min_demand;
				double randomValue_stdev = rng.genUniform();
				double p_dummy = 0.2;
				int64_t n = static_cast<int64_t>(std::round(state.mean_demand / p_dummy));
				double prob = state.mean_demand / n;
				double var = n * prob * (1 - prob);
				double binom_stdev = std::sqrt(var);
				state.stdev_demand = randomValue_stdev * (state.mean_demand * 2 - binom_stdev) + binom_stdev;
			}

			DynaPlex::DiscreteDist state_demand_dist = DiscreteDist::GetAdanEenigeResingDist(state.mean_demand, state.stdev_demand);
			state.min_demand = state_demand_dist.Min();
			state.max_demand = state_demand_dist.Max();
			state.demand_probs.reserve(state_demand_dist.DistinctValueCount());
			std::vector<double> demand_probs;
			for (const auto& [qty, prob] : state_demand_dist) {
				state.demand_probs.push_back(prob);
			}

			auto DemOverLeadtime = DiscreteDist::GetZeroDist();
			for (size_t i = 0; i <= state.leadtime; i++)
			{
				DemOverLeadtime = DemOverLeadtime.Add(state_demand_dist);
			}
			state.MaxOrderSize = state_demand_dist.Fractile(state.p / (state.p + h));
			state.MaxSystemInv = DemOverLeadtime.Fractile(state.p / (state.p + h));			

			auto queue = Queue<int64_t>{};
			queue.reserve(state.leadtime + 1);
			queue.push_back(0);//<- initial on-hand
			for (size_t i = 0; i < state.leadtime - 1; i++)
			{
				queue.push_back(0);
			}
			state.cat = StateCategory::AwaitAction();
			state.state_vector = queue;
			state.total_inv = queue.sum();
			return state;
		}
	
		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("state_vector", state.state_vector);
			vars.Get("mean_demand", state.mean_demand);
			vars.Get("demand_probs", state.demand_probs);
			vars.Get("max_demand", state.max_demand);
			vars.Get("min_demand", state.min_demand);
			vars.Get("stdev_demand", state.stdev_demand);
			vars.Get("total_inv", state.total_inv);
			vars.Get("p", state.p);
			vars.Get("leadtime", state.leadtime);
			vars.Get("MaxSystemInv", state.MaxSystemInv);
			vars.Get("MaxOrderSize", state.MaxOrderSize);
			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("demand_probs", demand_probs);
			vars.Add("max_demand", max_demand);
			vars.Add("min_demand", min_demand);
			vars.Add("state_vector", state_vector);
			vars.Add("mean_demand", mean_demand);
			vars.Add("stdev_demand", stdev_demand);
			vars.Add("total_inv", total_inv);
			vars.Add("p", p);
			vars.Add("leadtime", leadtime);
			vars.Add("MaxSystemInv", MaxSystemInv);
			vars.Add("MaxOrderSize", MaxOrderSize);
			return vars;
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "lost_sales_one_network_extended",
				/*description*/ "Canonical lost sales problem, see e.g. Zipkin (2008) for a formal description",
				/*reference to passed registry*/registry); 
		}
	}
}

