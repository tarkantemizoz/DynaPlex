#include "dynaplex/vargroup.h"
#include "dynaplex/error.h"
#include <gtest/gtest.h>
#include "dynaplex/dynaplexprovider.h"
#include "testutils.h" // for ExecuteTest

namespace DynaPlex::Tests {

	// Simplified stationary Super-MDP: deterministic lead time, non-cyclic demand.
	// State serialization is intentionally not simulation-complete (the demand estimator and
	// derived order limits are hidden state), so SkipStateSerializationTests stays true, as in
	// the full Zero_Shot_Lost_Sales_Inventory_Control tests.

	TEST(zero_shot_lost_sales_stationary, eval_deterministic_leadtime) {
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.SkipStateSerializationTests = true;
		tester.ExecuteTest("Zero_Shot_Lost_Sales_Stationary", "mdp_config_0.json");
	}

	TEST(zero_shot_lost_sales_stationary, eval_censored_demand_kaplan_meier) {
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.SkipStateSerializationTests = true;
		tester.ExecuteTest("Zero_Shot_Lost_Sales_Stationary", "mdp_config_1.json");
	}

	TEST(zero_shot_lost_sales_stationary, train_super_mdp) {
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.SkipStateSerializationTests = true;
		tester.ExecuteTest("Zero_Shot_Lost_Sales_Stationary", "mdp_config_2.json");
	}

	// Censoring is an evaluate-only (deployment-time) mechanism: a training config that also asks
	// for censored demand is contradictory and must be rejected, guaranteeing !evaluate => !censoredDemand.
	TEST(zero_shot_lost_sales_stationary, training_with_censored_demand_is_rejected) {
		auto& dp = DynaPlexProvider::Get();
		DynaPlex::VarGroup c;
		c.Add("id", "Zero_Shot_Lost_Sales_Stationary");
		c.Add("evaluate", false);
		c.Add("discount_factor", 1.0);
		c.Add("max_demand", 12.0);
		c.Add("max_p", 99.0);
		c.Add("max_leadtime", 10);
		c.Add("censoredDemand", true);
		ASSERT_THROW(dp.GetMDP(c), DynaPlex::Error);
	}

	// Mirror of the oracle base-stock guard for the full model: the base-stock benchmark may
	// order up past the per-instance OrderConstraint (the NN action-mask bound), and that must
	// not throw, since ModifyStateWithAction does not enforce IsAllowedAction.
	TEST(zero_shot_lost_sales_stationary, oracle_base_stock_may_exceed_order_constraint) {
		auto& dp = DynaPlexProvider::Get();
		DynaPlex::VarGroup c;
		c.Add("id", "Zero_Shot_Lost_Sales_Stationary");
		c.Add("evaluate", true);
		c.Add("discount_factor", 1.0);
		c.Add("max_demand", 12.0);
		c.Add("max_p", 99.0);
		c.Add("max_leadtime", 10);
		c.Add("p", 39.0);
		c.Add("leadtime", 6);
		c.Add("mean_demand", 8.0);
		c.Add("stdDemand", 2.8284271);

		DynaPlex::MDP mdp;
		ASSERT_NO_THROW(mdp = dp.GetMDP(c));

		DynaPlex::VarGroup test_config;
		test_config.Add("warmup_periods", 100);
		test_config.Add("number_of_trajectories", 50);
		test_config.Add("periods_per_trajectory", 500);
		test_config.Add("rng_seed", 1122);
		auto comparer = dp.GetPolicyComparer(mdp, test_config);

		DynaPlex::VarGroup pol;
		pol.Add("id", "base_stock");
		pol.Add("base_stock_level", 60); // exceeds a typical per-instance OrderConstraint
		ASSERT_NO_THROW(comparer.Assess(mdp->GetPolicy(pol)));
	}
}
