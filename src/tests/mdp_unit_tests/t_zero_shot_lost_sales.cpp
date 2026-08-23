#include "dynaplex/vargroup.h"
#include "dynaplex/error.h"
#include <gtest/gtest.h>
#include "dynaplex/dynaplexprovider.h"
#include "testutils.h" // for ExecuteTest

namespace DynaPlex::Tests {

	// Note on flags:
	// - SkipStateSerializationTests: this MDP's ToVarGroup/GetState is feature-complete
	//   (enough to reproduce the NN input) but not simulation-complete (the demand/lead-time
	//   estimators and derived limits are hidden state), so a serialize->deserialize->step
	//   round-trip is intentionally not reproducible. See serialization discussion in review.
	// - TestEventProbs stays false: this MDP does not expose exact event probabilities.

	TEST(zero_shot_lost_sales, eval_deterministic_leadtime_cyclic) {
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.SkipStateSerializationTests = true;
		tester.ExecuteTest("Zero_Shot_Lost_Sales_Inventory_Control", "mdp_config_0.json");
	}

	TEST(zero_shot_lost_sales, eval_stochastic_leadtime_order_crossover) {
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.SkipStateSerializationTests = true;
		tester.ExecuteTest("Zero_Shot_Lost_Sales_Inventory_Control", "mdp_config_1.json");
	}

	TEST(zero_shot_lost_sales, eval_censored_demand_kaplan_meier) {
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.SkipStateSerializationTests = true;
		tester.ExecuteTest("Zero_Shot_Lost_Sales_Inventory_Control", "mdp_config_2.json");
	}

	TEST(zero_shot_lost_sales, train_super_mdp) {
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.SkipStateSerializationTests = true;
		tester.ExecuteTest("Zero_Shot_Lost_Sales_Inventory_Control", "mdp_config_3.json");
	}

	TEST(zero_shot_lost_sales, eval_censored_leadtime_no_crossover) {
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.SkipStateSerializationTests = true;
		tester.ExecuteTest("Zero_Shot_Lost_Sales_Inventory_Control", "mdp_config_4.json");
	}

	TEST(zero_shot_lost_sales, eval_censored_leadtime_order_crossover) {
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.SkipStateSerializationTests = true;
		tester.ExecuteTest("Zero_Shot_Lost_Sales_Inventory_Control", "mdp_config_5.json");
	}

	// Oracle base-stock benchmark orders up to a target level and may exceed the per-instance
	// OrderConstraint (which is the NN action-mask bound, not a hard feasibility limit).
	// This must run without throwing - it guards against re-introducing an IsAllowedAction
	// guard in ModifyStateWithAction that would break the paper's base-stock comparisons.
	TEST(zero_shot_lost_sales, oracle_base_stock_may_exceed_order_constraint) {
		auto& dp = DynaPlexProvider::Get();
		DynaPlex::VarGroup c;
		c.Add("id", "Zero_Shot_Lost_Sales_Inventory_Control");
		c.Add("evaluate", true);
		c.Add("train_stochastic_leadtimes", true);
		c.Add("train_cyclic_demand", true);
		c.Add("discount_factor", 1.0);
		c.Add("max_demand", 12.0);
		c.Add("max_p", 100.0);
		c.Add("max_leadtime", 10);
		c.Add("max_num_cycles", 7);
		c.Add("p", 39.0);
		c.Add("leadtime", 6);
		std::vector<int64_t> demand_cycles{ 0, 1, 2 };
		std::vector<double> mean_demand{ 8.0, 10.0, 6.0 };
		std::vector<double> std_demand{ 2.8284271, 3.1622777, 2.4494897 };
		c.Add("demand_cycles", demand_cycles);
		c.Add("mean_demand", mean_demand);
		c.Add("stdDemand", std_demand);

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
		pol.Add("base_stock_level", 60); // exceeds a typical per-instance OrderConstraint (~14)
		ASSERT_NO_THROW(comparer.Assess(mdp->GetPolicy(pol)));
	}
}
