#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace random_leadtimes
	{
		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;		
			vars.Add("valid_actions", MaxOrderSize + 1);		
			vars.Add("discount_factor", discount_factor);
			return vars;
		}

		MDP::MDP(const VarGroup& config)
		{
			config.Get("b", b);
			config.Get("h", h);
			config.Get("DemandRate", DemandRate);
			config.Get("InitialInventoryLevel", InitialInventoryLevel);
			config.Get("InitialOrdersInPipeline", InitialOrdersInPipeline);
			config.Get("lead_time_dist", lead_time_dist);
			config.Get("optimized", optimized);
			if (config.HasKey("optimized"))
				config.Get("optimized", optimized);
			else
				optimized = false;
			if (config.HasKey("BaseStockLevel"))
				config.Get("BaseStockLevel", BaseStockLevel);
			exponential_lead_time_dist = false;

			if (lead_time_dist == "exponential") {
				config.Get("ExponentialLeadTimeRate", ExponentialLeadTimeRate);
				exponential_lead_time_dist = true;
			}
			else if (lead_time_dist == "uniform") {
				config.Get("UniformStart", UniformStart);
				config.Get("UniformEnd", UniformEnd);
			}
			else if (lead_time_dist == "pareto") {
				config.Get("q", q);
				config.Get("tau", tau);
			}
			else {
				throw DynaPlex::Error("MDP instance: lead time distribution not supported.");
			}

			if (optimized) {
				Kmax = 2.0;
				Kmin = 2.0;
				Kbo = 10.0;
				MaxInventoryPosition = BaseStockLevel + static_cast<int64_t>(std::ceil(Kmax * std::sqrt(static_cast<double>(BaseStockLevel))));
				MaxAllowedBackorder = static_cast<int64_t>(std::ceil(Kbo * std::sqrt(static_cast<double>(BaseStockLevel))));
				MaxOrdersInPipeline = MaxInventoryPosition + MaxAllowedBackorder;
				MinInventoryPosition = std::max((int64_t)0, BaseStockLevel - static_cast<int64_t>(std::ceil(Kmin * std::sqrt(static_cast<double>(BaseStockLevel)))));
			}
			else {
				MaxInventoryPosition = 100; // should be sufficiently high when optimizing base stock level
				MaxAllowedBackorder = 100; // should be sufficiently high when optimizing base stock level
				MaxOrdersInPipeline = MaxInventoryPosition + MaxAllowedBackorder;
				MinInventoryPosition = 0;
			}
			MaxOrderSize = 6;

			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const 
		{
			if ((state.OrdersInPipeline + action <= MaxOrdersInPipeline) && (state.InventoryPosition + action <= MaxInventoryPosition) && (state.InventoryPosition + action >= MinInventoryPosition))
			{
				return true;
			}
			if ((state.OrdersInPipeline + action <= MaxOrdersInPipeline) && (state.InventoryPosition + action <= MaxInventoryPosition) && (state.InventoryPosition + action < MinInventoryPosition))
			{
				return action == MaxOrderSize;
			}
			if ((state.OrdersInPipeline + action <= MaxOrdersInPipeline) && (state.InventoryPosition + action > MaxInventoryPosition) && (state.InventoryPosition + action >= MinInventoryPosition))
			{
				return action == (int64_t)0;
			}
			return false;
		}

		double MDP::ModifyStateWithAction(State& state, int64_t action) const
		{
			state.OrdersInPipeline += action;
			state.InventoryPosition += action;
			
			if (!exponential_lead_time_dist) {
				for (size_t i = 0; i < action; i++)
				{
					state.OrderTimes.push_back(0.0);
				}
			}

			state.cat = StateCategory::AwaitEvent();
			return 0.0;
		}

		MDP::Event MDP::GetEvent(const State& state, RNG& rng) const 
		{
			double dE = rng.genUniform();
			double NextDemandTime = std::log(1 - dE) / (-DemandRate);

			std::vector<double> NextOrderTime;
			NextOrderTime.reserve(state.OrdersInPipeline);
			for (size_t i = 0; i < state.OrdersInPipeline; i++)
			{
				NextOrderTime.push_back(rng.genUniform());
			}
			for (size_t i = state.OrdersInPipeline; i < MaxOrdersInPipeline; i++)
			{
				rng.genUniform();
			}

			return { NextDemandTime, NextOrderTime };
		}

		double MDP::ModifyStateWithEvent(State& state,const MDP::Event& event) const
		{
			state.cat = StateCategory::AwaitAction();

			std::vector<double> ReceivedOrderTimesInThisPeriod{};
			std::vector<size_t> ReceivedOrders{};
			double prob = 0.0;
			if (lead_time_dist == "exponential")
				prob = 1 - std::exp(-ExponentialLeadTimeRate * event.first);
			for (size_t i = 0; i < state.OrdersInPipeline; i++)
			{
				if (lead_time_dist == "uniform")
					prob = event.first / (UniformEnd - state.OrderTimes[i]);
				else if (lead_time_dist == "pareto")
					prob = 1.0 - pow((1.0 + tau * state.OrderTimes[i]) / (1.0 + tau * (state.OrderTimes[i] + event.first)), q);

				if (event.second[i] <= prob)
				{
					double orderArrivalTime = 0.0;
					if (exponential_lead_time_dist) {
						orderArrivalTime = std::log(1 - event.second[i]) / (-ExponentialLeadTimeRate);
					}
					else {
						orderArrivalTime = event.second[i] * event.first / prob;
						ReceivedOrders.push_back(i);
					}
					ReceivedOrderTimesInThisPeriod.push_back(orderArrivalTime);						
				}
				else if (!exponential_lead_time_dist)
				{
					state.OrderTimes[i] += event.first;
				}
			}

			if (ReceivedOrderTimesInThisPeriod.size() > 1)
			{
				std::sort(ReceivedOrderTimesInThisPeriod.begin(), ReceivedOrderTimesInThisPeriod.end());
				if (!exponential_lead_time_dist)
					std::sort(ReceivedOrders.begin(), ReceivedOrders.end(), std::greater<size_t>());
			}

			if (!exponential_lead_time_dist) {
				for (size_t i = 0; i < ReceivedOrders.size(); i++)
				{
					state.OrderTimes.erase(state.OrderTimes.begin() + ReceivedOrders[i]);
				}
			}

			double Costs{ 0.0 };
			double LastComingOrderTime{ 0.0 };
			for (int64_t i = 0; i < ReceivedOrderTimesInThisPeriod.size(); i++)
			{
				double time = ReceivedOrderTimesInThisPeriod[i] - LastComingOrderTime;
				LastComingOrderTime = ReceivedOrderTimesInThisPeriod[i];
				if (state.InventoryLevel < 0)
					Costs += -state.InventoryLevel * b * time;
				else
					Costs += state.InventoryLevel * h * time;
				state.InventoryLevel += 1;
				state.OrdersInPipeline -= 1;
			}

			if (state.InventoryLevel < 0)
				Costs += -state.InventoryLevel * b * (event.first - LastComingOrderTime);		
			else
				Costs += state.InventoryLevel * h * (event.first - LastComingOrderTime);			
			state.InventoryLevel -= 1;
			state.InventoryPosition -= 1;

			if (optimized) {
				if (state.InventoryLevel < 0 && MaxAllowedBackorder + state.InventoryLevel < 0)
				{
					std::cout << "Extreme case, demand is lost not backlogged, but lost. -- Increasing MaxAllowedBackorder might be beneficial." << std::endl;
					state.InventoryLevel += 1;
					state.InventoryPosition += 1;
				}
			}
			
			return Costs * DemandRate;
		}
			
		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			features.Add(state.InventoryLevel);
			features.Add(state.OrdersInPipeline);
			if (!exponential_lead_time_dist) {
				std::vector<double> OrderInPipelineTimes = state.OrderTimes;
				std::sort(OrderInPipelineTimes.begin(), OrderInPipelineTimes.end(), std::greater<double>());
				features.Add(OrderInPipelineTimes);
				std::vector<double> PaddedVec(MaxOrdersInPipeline - OrderInPipelineTimes.size(), 0.0);
				features.Add(PaddedVec);
			}
		}

		MDP::State MDP::GetInitialState() const
		{
			State state{};

			state.InventoryLevel = InitialInventoryLevel;
			state.OrdersInPipeline = InitialOrdersInPipeline;
			state.InventoryPosition = state.InventoryLevel + state.OrdersInPipeline;
			for (size_t i = 0; i < InitialOrdersInPipeline; i++)
			{
				state.OrderTimes.push_back(0.0);
			}

			state.cat = StateCategory::AwaitAction();
			return state;
		}

		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("InventoryLevel", state.InventoryLevel);
			vars.Get("OrdersInPipeline", state.OrdersInPipeline);
			if (lead_time_dist != "exponential")
				vars.Get("OrderTimes", state.OrderTimes);
			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("InventoryLevel", InventoryLevel);
			vars.Add("OrdersInPipeline", OrdersInPipeline);
			vars.Add("OrderTimes", OrderTimes);
			return vars;
		}
		
		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{
			registry.Register<BaseStockPolicy>("base_stock",
				"Base-stock policy with parameter base_stock_level.");
			registry.Register<InitialPolicy>("initial_policy",
				"Base-stock policy with adjusted conditions to adhere IsAllowedAction.");
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "random_leadtimes",
				/*description*/ "Inventory systems with random leadtimes and backlog, see e.g. Temizoz (2023) for a formal description.",
				/*reference to passed registry*/registry); 
		}
	}
}

