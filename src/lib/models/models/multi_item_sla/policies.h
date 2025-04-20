#pragma once
#include <cstdint>
#include "mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Models {
	namespace multi_item_sla 
	{
		class MDP;

		class BaseStockPolicy
		{
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			int64_t serviceLevelPolicy;
		public:
			BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		class GreedyDynamicPolicy
		{
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			int64_t serviceLevelPolicy;
		public:
			GreedyDynamicPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		class DynamicPolicy
		{
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			double bestPolicyFillRate;
			std::vector<double> policyFillRates;
		public:
			DynamicPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};
	}
}

