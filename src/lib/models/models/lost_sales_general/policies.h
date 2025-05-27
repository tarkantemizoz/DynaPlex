#pragma once
#include <cstdint>
#include "mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>
#include "dynaplex/modelling/discretedist.h"

namespace DynaPlex::Models {
	namespace lost_sales_general /*must be consistent everywhere for complete mdp defininition and associated policies.*/
	{
		class MDP;

		class GreedyCappedBaseStockPolicy
		{
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
		public:
			GreedyCappedBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};
	}
}

