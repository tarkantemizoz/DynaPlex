﻿#include "dynaplex/vargroup.h"
#include "dynaplex/error.h"
#include <gtest/gtest.h>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/trajectory.h"
#include "dynaplex/demonstrator.h"
#include "testutils.h" // for ExecuteTest
namespace DynaPlex::Tests {
	
	TEST(lost_sales_one_network, mdp_config_1_policy_config_1) {
		std::string model_name = "lost_sales_one_network";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_1.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_1.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.TestEventProbs = false;
		tester.SkipEqualityTests = true;
		tester.SkipStateSerializationTests = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(lost_sales_one_network, mdp_config_1_policy_config_2) {
		std::string model_name = "lost_sales_one_network";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_1.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_2.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.TestEventProbs = false;
		tester.SkipEqualityTests = true;
		tester.SkipStateSerializationTests = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(lost_sales_one_network, mdp_config_2_policy_config_1) {
		std::string model_name = "lost_sales_one_network";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_2.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_1.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.TestEventProbs = false;
		tester.SkipEqualityTests = true;
		tester.SkipStateSerializationTests = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(lost_sales_one_network, mdp_config_2_policy_config_2) {
		std::string model_name = "lost_sales_one_network";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_2.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_2.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		tester.TestEventProbs = false;
		tester.SkipEqualityTests = true;
		tester.SkipStateSerializationTests = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(lost_sales_one_network, Basics) {
		auto& dp = DynaPlexProvider::Get();
		DynaPlex::VarGroup vars;
		vars.Add("id", "lost_sales_one_network");
		vars.Add("max_p", 50.0);
		vars.Add("max_leadtime", 10);
		vars.Add("evaluate", false);
		vars.Add("p", 4.0);
		vars.Add("leadtime", 3);
		vars.Add("discount_factor", 1.0);
		vars.Add("mean_demand", 3.0);
		vars.Add("stdev_demand", 3.0);

		DynaPlex::MDP mdp;
		DynaPlex::Policy policy;

		ASSERT_NO_THROW(
			mdp = dp.GetMDP(vars);
		);
		ASSERT_NO_THROW(
			policy = mdp->GetPolicy("random");
		);

		Trajectory trajectory{};



		ASSERT_NO_THROW(
			trajectory.RNGProvider.SeedEventStreams(true, 123);
		);
		ASSERT_NO_THROW(
			mdp->InitiateState({ &trajectory,1 });
		);


	    int64_t max_period_count = 10;
		bool finalreached = false;
		while (trajectory.PeriodCount < max_period_count && !finalreached)
		{
			auto& cat = trajectory.Category;
			if (cat.IsAwaitEvent())
			{
				ASSERT_NO_THROW(
					mdp->IncorporateEvent({ &trajectory,1 });
				);
			}
			else if (cat.IsAwaitAction())
			{
				ASSERT_NO_THROW(
					policy->SetAction({ &trajectory,1 });
				);
				ASSERT_NO_THROW(
					mdp->IncorporateAction({ &trajectory,1 });
				);
			}
			else if (cat.IsFinal())
			{
				finalreached = true;
			}
		}			
	}
}