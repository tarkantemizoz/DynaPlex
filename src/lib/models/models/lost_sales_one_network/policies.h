#pragma once
#include <cstdint>
#include "mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Models {
	namespace lost_sales_one_network
	{
		// Forward declaration
		class MDP;

		class BaseStockPolicy
		{
			//this is the MDP defined inside the current namespace!
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			int64_t base_stock_level;
		public:
			BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		class CappedBaseStockPolicy
		{
			std::shared_ptr<const MDP> mdp;
			int64_t base_stock_level;
			int64_t cap;
		public:
			CappedBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		class GreedyCappedBaseStockPolicy
		{
			std::shared_ptr<const MDP> mdp;
		public:
			GreedyCappedBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};
	}
}

