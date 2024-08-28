#include <gtest/gtest.h>
#include "dynaplex/dynaplexprovider.h"
using namespace DynaPlex;

namespace DynaPlex::Tests {
	TEST(ExactAlgorithm, LostSalesSmallInstances) {
		auto& dp = DynaPlexProvider::Get();

		DynaPlex::VarGroup config;
		//retrieve MDP registered under the id string "lost_sales":
		config.Add("id", "lost_sales");
		config.Add("h", 1.0);
		config.Add("p", 4.0);
		config.Add("leadtime", 2);
		config.Add("demand_dist", DynaPlex::VarGroup({
			{"type", "poisson"},
			{"mean", 5.0}
			}));
		//This to ensure that the base-stock policy actually is free to order as it likes
		config.Add("bound_order_size_to_max_system_inv", true);

		DynaPlex::MDP mdp;
		DynaPlex::Policy policy;
		DynaPlex::VarGroup exact_config = DynaPlex::VarGroup{ {"max_states",100000}, {"silent", true } };

		ASSERT_NO_THROW({ mdp = dp.GetMDP(config); });
		ASSERT_NO_THROW({ policy = mdp->GetPolicy("base_stock"); });
		{
			auto ExactSolver = dp.GetExactSolver(mdp, exact_config);
			double bs_costs, opt_costs;
			ASSERT_NO_THROW({ bs_costs = ExactSolver.ComputeCosts(policy); });
			ASSERT_NO_THROW({ opt_costs = ExactSolver.ComputeCosts(); });

			//from Zipkin paper (pois, L=2, p=9):
			EXPECT_NEAR(bs_costs, 4.94, 0.005);
			EXPECT_NEAR(opt_costs, 4.40, 0.005);

			auto ExactPolicy = ExactSolver.GetOptimalPolicy();

			auto comparer = dp.GetPolicyComparer(mdp);
			auto return_val = comparer.Compare(policy, ExactPolicy);
			return_val[0].Get("mean", bs_costs);
			return_val[1].Get("mean", opt_costs);

			EXPECT_NEAR(bs_costs, 4.94, 0.015);
			EXPECT_NEAR(opt_costs, 4.40, 0.015);

		}
		config.Set("leadtime", 1);
		{
			mdp = dp.GetMDP(config);
			policy = mdp->GetPolicy("base_stock");
			auto ExactSolver = dp.GetExactSolver(mdp, exact_config);
			double bs_costs, opt_costs;
			ASSERT_NO_THROW({ bs_costs = ExactSolver.ComputeCosts(policy); });
			ASSERT_NO_THROW({ opt_costs = ExactSolver.ComputeCosts(); });

			//from Zipkin paper:(pois, L=1, p=4):
			EXPECT_NEAR(bs_costs, 4.39, 0.005);
			EXPECT_NEAR(opt_costs, 4.04, 0.01);
		}
	}
}