#pragma once
#include <cstdint>
#include "mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Models {
	namespace lost_sales_one_network_extended
	{
		// Forward declaration
		class MDP;

		class ConstantOrderPolicy
		{
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			int64_t co_level;
		public:
			ConstantOrderPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

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
			const VarGroup varGroup;
			int64_t S;
			int64_t r;
		public:
			CappedBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

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

