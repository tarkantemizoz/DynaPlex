#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace perishable_systems /*keep this in line with id below and with namespace name in header*/
	{
		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;
			vars.Add("valid_actions", MaxSystemInv + 1);
			vars.Add("discount_factor", discount_factor);

			VarGroup diagnostics{};
			diagnostics.Add("MaxSystemInv", MaxSystemInv);
			vars.Add("diagnostics", diagnostics);

			return vars;
		}

		double MDP::ModifyStateWithAction(State& state, int64_t action) const
		{
			double cost = c * action;
			state.state_vector.push_back(action);
			state.total_inv += action;
			state.cat = StateCategory::AwaitEvent();

			return cost;
		}

		MDP::MDP(const VarGroup& config)
		{
			config.GetOrDefault("o", o, 100.0);
			config.GetOrDefault("c", c, 0.0);
			config.GetOrDefault("h", h, 0.0);
			config.GetOrDefault("p", p, o);
			config.GetOrDefault("mu", mu, 4.0);

			config.Get("f", f);
			config.Get("cvr", cvr);
			config.Get("LeadTime", LeadTime);
			config.Get("ProductLife", ProductLife);
			config.Get("enable_seq_halving", enable_seq_halving);

			double st_dev = cvr * std::sqrt(mu);

			//providing discount_factor is optional. 
			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;

			DynaPlex::DiscreteDist fifo_demand_dist;
			DynaPlex::DiscreteDist lifo_demand_dist;
			//Initiate members that are computed from the parameters:
			double FIFOmu = f * mu;
			if (FIFOmu > 0) {
				fifo_demand_dist = DiscreteDist::GetAdanEenigeResingDist(FIFOmu, std::sqrt(f) * st_dev);
			}
			else {
				fifo_demand_dist = DiscreteDist::GetZeroDist();
			}

			double LIFOmu = (1 - f) * mu;
			if (LIFOmu > 0) {
				lifo_demand_dist = DiscreteDist::GetAdanEenigeResingDist(LIFOmu, std::sqrt(1 - f) * st_dev);
			}
			else {
				lifo_demand_dist = DiscreteDist::GetZeroDist();
			}

			auto DemOverLeadtime = DiscreteDist::GetZeroDist();
			for (size_t i = 0; i <= LeadTime + ProductLife; i++)
			{
				DemOverLeadtime = DemOverLeadtime.Add(fifo_demand_dist);
				DemOverLeadtime = DemOverLeadtime.Add(lifo_demand_dist);
			}
			MaxSystemInv = DemOverLeadtime.Fractile(p / (p + o));

			// also possible to use this
			//std::vector<DiscreteDist> dist = { fifo_demand_dist, lifo_demand_dist };
			//demand_dist = JointDiscreteDist(dist);

			demand_dist = JointDiscreteDist(fifo_demand_dist, lifo_demand_dist);
			demand_combination_holder = demand_dist.GetJointQtys();
		}

		MDP::Event MDP::GetEvent(RNG& rng) const {
			return demand_dist.GetSample(rng);
		}

		std::vector<std::tuple<MDP::Event, double>> MDP::EventProbabilities() const {
			return demand_dist.QuantityProbabilities();
		}

		double MDP::ModifyStateWithEvent(State& state, const MDP::Event& event) const
		{
			state.cat = StateCategory::AwaitAction();

			int64_t onHand = 0;
			for (size_t i = 0; i < ProductLife; i++) {
				onHand += state.state_vector.at(i);
			}
			int64_t FIFOdemand = demand_combination_holder[event][0];
			int64_t LIFOdemand = demand_combination_holder[event][1];
			int64_t TotalDemand = FIFOdemand + LIFOdemand;

			double cost = 0.0;
			// First check if there is enough on hand inventory
			if (onHand < TotalDemand) {
				cost += p * (TotalDemand - onHand);
				state.state_vector.pop_front();
				state.total_inv -= onHand;

				for (int64_t i = 0; i < ProductLife - 1; i++) {
					state.state_vector.at(i) = 0;
				}
			}
			else {
				// Meet fifo demand
				if (FIFOdemand > 0) {
					for (int64_t i = 0; i < ProductLife; i++) {
						int64_t inv = state.state_vector.at(i);
						if (inv > 0 && FIFOdemand > 0) {
							state.state_vector.at(i) = std::max((int64_t)0, inv - FIFOdemand);
							FIFOdemand = std::max((int64_t)0, FIFOdemand - inv);
						}
					}
				}
				if (LIFOdemand > 0) {
					for (int64_t i = ProductLife - 1; i >= 0; i--) {
						int64_t inv = state.state_vector.at(i);
						if (inv > 0 && LIFOdemand > 0) {
							state.state_vector.at(i) = std::max((int64_t)0, inv - LIFOdemand);
							LIFOdemand = std::max((int64_t)0, LIFOdemand - inv);
						}
					}
				}
				int64_t perishedInv = state.state_vector.pop_front();
				state.total_inv -= (perishedInv + TotalDemand);
				cost += o * perishedInv;
				cost += h * (onHand - TotalDemand);
			}

			return cost;
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const
		{
			features.Add(state.state_vector);
		}

		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{
			registry.Register<BaseStockPolicy>("base_stock",
				"Base-stock policy with parameter base_stock_level - default parameter is equal"
				" to the newsvendor solution required to cater to demand over ProductLife + Leadtime periods, i.e.,"
				" when the inventory on hand and in the pipeline have turned into waste");
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
			return (state.total_inv + action) <= MaxSystemInv || action == 0;
		}

		MDP::State MDP::GetInitialState() const
		{
			auto queue = Queue<int64_t>{};

			queue.reserve(LeadTime + ProductLife);
			for (size_t i = 0; i < LeadTime + ProductLife - 1; i++)
			{
				queue.push_back(0);
			}
			State state{};
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
			vars.Get("total_inv", state.total_inv);
			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("state_vector", state_vector);
			vars.Add("total_inv", total_inv);
			return vars;
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "perishable_systems",
				/*description*/ "Perishable inventory systems, see e.g. Temizoz et al. (2023) and Haijema and Minner (2019) for a formal description.",
				/*reference to passed registry*/registry);
		}

	}
}