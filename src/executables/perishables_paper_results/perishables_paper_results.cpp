#include <iostream>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

//Find the best base-stock level
int64_t FindBestBSLevel(DynaPlex::VarGroup& config)
{
	auto& dp = DynaPlexProvider::Get();
	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 100);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);

	auto comparer = dp.GetPolicyComparer(mdp, test_config);
	double bestBScost = std::numeric_limits<double>::infinity();
	int64_t BSLevel = 1;
	int64_t bestBSLevel = BSLevel;

	DynaPlex::VarGroup policy_config;
	policy_config.Add("id", "base_stock");
	policy_config.Add("base_stock_level", BSLevel);

	while (true)
	{
		auto policy = mdp->GetPolicy(policy_config);
		auto comparison = comparer.Assess(policy);
		double cost;
		comparison.Get("mean", cost);
		if (cost < bestBScost)
		{
			bestBScost = cost;
			bestBSLevel = BSLevel;
			BSLevel++;
			policy_config.Set("base_stock_level", BSLevel);
		}
		else {
			break;
		}
	}

	return bestBSLevel;
}

// Test of Temizoz et al. (2023) instances
void TestPaperInstances()
{
	auto& dp = DynaPlexProvider::Get();

	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",15},
		{"mini_batch_size", 64},
		{"max_training_epochs", 1000},
		{"train_based_on_probs", false}
	};

	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,128}}
	};

	int64_t num_gens = 3;
	bool enable_seq_halving = true;
	DynaPlex::VarGroup dcl_config{
		//use paper hyperparameters everywhere. 
		{"N",5000},
		{"num_gens",num_gens},
		{"M",1000},
		{"H", 40},
		{"L", 100},
		{"nn_architecture",nn_architecture},
		{"enable_sequential_halving", enable_seq_halving},
		{"nn_training",nn_training}
	};

	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 100);
	test_config.Add("number_of_trajectories", 1000);
	test_config.Add("periods_per_trajectory", 5000);
	test_config.Add("rng_seed", 1122);

	DynaPlex::VarGroup exact_config = DynaPlex::VarGroup{ {"max_states",10000000}, {"silent", true } };

	std::vector<std::vector<double>> ShelfLifeResultsSmall(3);
	std::vector<std::vector<double>> LeadTimeResultsSmall(3);
	std::vector<std::vector<double>> CVRResultsSmall(3);
	std::vector<std::vector<double>> fResultsSmall(3);
	std::vector<double> AllResultsSmall;

	std::vector<std::vector<std::vector<double>>> ShelfLifeResultsLarge(3);
	std::vector<std::vector<std::vector<double>>> LeadTimeResultsLarge(3);
	std::vector<std::vector<std::vector<double>>> CVRResultsLarge(3);
	std::vector<std::vector<std::vector<double>>> fResultsLarge(3);
	std::vector<std::vector<double>> AllResultsLarge;

	std::vector<std::vector<std::vector<double>>> ShelfLifeResults(3);
	std::vector<std::vector<std::vector<double>>> LeadTimeResults(3);
	std::vector<std::vector<std::vector<double>>> CVRResults(3);
	std::vector<std::vector<std::vector<double>>> fResults(3);
	std::vector<std::vector<double>> AllResults;

	int64_t opt_results_index = 0;
	int64_t ShelfLifeIndex = 0;
	std::string id = "perishable_systems";
	for (int64_t m : {3, 4, 5})
	{
		int64_t LeadTimeIndex = 0;
		for (int64_t leadtime : {0, 1, 2})
		{
			int64_t CVRindex = 0;
			for (double cvr : { 1.0, 1.5, 2.0 })
			{
				int64_t findex = 0;
				for (double f : {1.0, 0.5, 0.0})
				{
					DynaPlex::VarGroup config;
					config.Add("id", id);
					config.Add("f", f);
					config.Add("cvr", cvr);
					config.Add("ProductLife", m);
					config.Add("LeadTime", leadtime);
					config.Add("enable_seq_halving", enable_seq_halving);

					int64_t BestBSLevel = FindBestBSLevel(config);
					DynaPlex::VarGroup policy_config;
					policy_config.Add("id", "base_stock");
					policy_config.Add("base_stock_level", BestBSLevel);

					DynaPlex::MDP mdp = dp.GetMDP(config);
					auto policy = mdp->GetPolicy(policy_config);
					auto dcl = dp.GetDCL(mdp, policy, dcl_config);
					dcl.TrainPolicy();

					std::string life = "_m" + std::to_string(m);
					std::string leadtime_str = "_l" + std::to_string(leadtime);
					std::string cvr_str = "_l" + std::to_string(cvr);
					std::string f_str = "_f" + std::to_string(f);

					std::string loc = id + life + leadtime_str + cvr_str + f_str;
					if (!enable_seq_halving) {
						loc = loc + "_DCL0";
					}
					dp.System() << "Network id:  " << loc << std::endl;

					for (int64_t gen = 1; gen <= num_gens; gen++)
					{
						auto policy = dcl.GetPolicy(gen);
						auto path = dp.System().filepath(loc, "dcl_gen" + gen);
						dp.SavePolicy(policy, path);
					}

					auto policies = dcl.GetPolicies();

					std::vector<DynaPlex::Policy> policies;
					policies.push_back(policy);
					for (int64_t gen = 1; gen <= num_gens; gen++)
					{
						auto path = dp.System().filepath(loc, "dcl_gen" + gen);
						auto nn_policy = dp.LoadPolicy(mdp, path);
						policies.push_back(nn_policy);
					}

					std::cout << std::endl;
					std::cout << config.Dump() << std::endl;
					std::vector<double> results{};
					double best_nn_cost = std::numeric_limits<double>::infinity();
					if (!((m + leadtime <= 5 && (f == 0.0 || f == 1.0)) || (m + leadtime <= 4 && f == 0.5))) {
						auto comparer = dp.GetPolicyComparer(mdp, test_config);
						auto comparison = comparer.Compare(policies);

						double best_bs_cost{ 0.0 };
						for (auto& VarGroup : comparison)
						{
							std::cout << VarGroup.Dump() << std::endl;
							DynaPlex::VarGroup policy_id;
							VarGroup.Get("policy", policy_id);
							std::string id;
							policy_id.Get("id", id);

							if (id == "NN_Policy") {
								double nn_cost;
								VarGroup.Get("mean", nn_cost);
								if (nn_cost < best_nn_cost) {
									best_nn_cost = nn_cost;
								}
							}
							else if (id == "base_stock") {
								VarGroup.Get("mean", best_bs_cost);
							}
						}

						double BSNNGap = (100 * (best_nn_cost - best_bs_cost) / best_bs_cost);
						std::cout << std::endl;
						std::cout << "Best base-stock policy cost:  " << best_bs_cost << "  best nn-policy cost:  " << best_nn_cost << "  gap:  " << BSNNGap << std::endl;
						std::cout << std::endl;

						results.push_back(best_bs_cost);
						results.push_back(best_nn_cost);
						results.push_back(BSNNGap);

						ShelfLifeResultsLarge[ShelfLifeIndex].push_back(results);
						LeadTimeResultsLarge[LeadTimeIndex].push_back(results);
						CVRResultsLarge[CVRindex].push_back(results);
						fResultsLarge[findex].push_back(results);
						AllResultsLarge.push_back(results);
					}
					else {
						auto exactsolver = dp.GetExactSolver(mdp, exact_config);
						double optimal_cost = exactsolver.ComputeCosts();
						double bs_pol_cost = exactsolver.ComputeCosts(policy);

						for (int64_t i = 1; i < policies.size(); i++) {
							auto nn_policy = policies[i];
							double nn_pol_cost = exactsolver.ComputeCosts(nn_policy);
							if (nn_pol_cost < best_nn_cost)
								best_nn_cost == nn_pol_cost;
						}
						double nn_opt_gap = 100 * (best_nn_cost - optimal_cost) / optimal_cost;
						double nn_bs_gap = 100 * (best_nn_cost - bs_pol_cost) / bs_pol_cost;

						std::cout << std::endl;
						std::cout << "Best base-stock policy cost:  " << bs_pol_cost << "  best nn-policy cost:  " << best_nn_cost << "  DCL BS gap:  " << nn_bs_gap << "  DCL opt gap:   " << nn_opt_gap << std::endl;
						std::cout << std::endl;

						ShelfLifeResultsSmall[ShelfLifeIndex].push_back(nn_opt_gap);
						LeadTimeResultsSmall[LeadTimeIndex].push_back(nn_opt_gap);
						CVRResultsSmall[CVRindex].push_back(nn_opt_gap);
						fResultsSmall[findex].push_back(nn_opt_gap);
						AllResultsSmall.push_back(nn_opt_gap);
						opt_results_index++;

						results.push_back(bs_pol_cost);
						results.push_back(best_nn_cost);
						results.push_back(nn_bs_gap);
					}
					ShelfLifeResults[ShelfLifeIndex].push_back(results);
					LeadTimeResults[LeadTimeIndex].push_back(results);
					CVRResults[CVRindex].push_back(results);
					fResults[findex].push_back(results);
					AllResults.push_back(results);

					findex++;
				}
				CVRindex++;
			}
			LeadTimeIndex++;
		}
		ShelfLifeIndex++;
	}

	std::cout << std::endl;
	//All results
	double BSCosts{ 0.0 };
	double NNCosts{ 0.0 };
	double BSNNGaps{ 0.0 };

	for (size_t i = 0; i < AllResults.size(); i++)
	{
		BSCosts += AllResults[i][0];
		NNCosts += AllResults[i][1];
		BSNNGaps += AllResults[i][2];
	}
	size_t TotalNumInstance = AllResults.size();
	std::cout << "Avg BS Costs:  " << BSCosts / TotalNumInstance << "  , Avg NN Costs:  " << NNCosts / TotalNumInstance;
	std::cout << "  , Avg BS - NN Gap:  " << BSNNGaps / TotalNumInstance << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

	//Shelf life results
	size_t mcount = 0;
	for (size_t m : {3, 4, 5})
	{
		double mBSCosts{ 0.0 };
		double mNNCosts{ 0.0 };
		double mBSNNGaps{ 0.0 };

		for (size_t i = 0; i < ShelfLifeResults[mcount].size(); i++)
		{
			mBSCosts += ShelfLifeResults[mcount][i][0];
			mNNCosts += ShelfLifeResults[mcount][i][1];
			mBSNNGaps += ShelfLifeResults[mcount][i][2];
		}
		size_t mTotalNumInstance = ShelfLifeResults[mcount].size();
		std::cout << "Product life:  " << m << "  Avg BS Costs:  " << mBSCosts / mTotalNumInstance;
		std::cout << "  , Avg NN Costs:  " << mNNCosts / mTotalNumInstance;
		std::cout << "  , Avg BS - NN Gap:  " << mBSNNGaps / mTotalNumInstance << std::endl;

		mcount++;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	//Lead time results
	size_t lcount = 0;
	for (size_t l : {0, 1, 2})
	{
		double mBSCosts{ 0.0 };
		double mNNCosts{ 0.0 };
		double mBSNNGaps{ 0.0 };

		for (size_t i = 0; i < LeadTimeResults[lcount].size(); i++)
		{
			mBSCosts += LeadTimeResults[lcount][i][0];
			mNNCosts += LeadTimeResults[lcount][i][1];
			mBSNNGaps += LeadTimeResults[lcount][i][2];
		}
		size_t mTotalNumInstance = LeadTimeResults[lcount].size();
		std::cout << "Lead time:  " << l << "  Avg BS Costs:  " << mBSCosts / mTotalNumInstance;
		std::cout << "  , Avg NN Costs:  " << mNNCosts / mTotalNumInstance;
		std::cout << "  , Avg BS - NN Gap:  " << mBSNNGaps / mTotalNumInstance << std::endl;

		lcount++;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	//CVR results
	size_t ccount = 0;
	for (double cvr : {1.0, 1.5, 2.0})
	{
		double mBSCosts{ 0.0 };
		double mNNCosts{ 0.0 };
		double mBSNNGaps{ 0.0 };

		for (size_t i = 0; i < CVRResults[ccount].size(); i++)
		{
			mBSCosts += CVRResults[ccount][i][0];
			mNNCosts += CVRResults[ccount][i][1];
			mBSNNGaps += CVRResults[ccount][i][2];
		}
		size_t mTotalNumInstance = CVRResults[ccount].size();
		std::cout << "CVR:  " << cvr << "  Avg BS Costs:  " << mBSCosts / mTotalNumInstance;
		std::cout << "  , Avg NN Costs:  " << mNNCosts / mTotalNumInstance;
		std::cout << "  , Avg BS - NN Gap:  " << mBSNNGaps / mTotalNumInstance << std::endl;

		ccount++;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	//f results
	size_t fcount = 0;
	for (double f : {1.0, 0.5, 0.0})
	{
		double mBSCosts{ 0.0 };
		double mNNCosts{ 0.0 };
		double mBSNNGaps{ 0.0 };

		for (size_t i = 0; i < fResults[fcount].size(); i++)
		{
			mBSCosts += fResults[fcount][i][0];
			mNNCosts += fResults[fcount][i][1];
			mBSNNGaps += fResults[fcount][i][2];
		}
		size_t mTotalNumInstance = fResults[fcount].size();
		std::cout << "Issuance Policy f:  " << f << "  Avg BS Costs:  " << mBSCosts / mTotalNumInstance;
		std::cout << "  , Avg NN Costs:  " << mNNCosts / mTotalNumInstance;
		std::cout << "  , Avg BS - NN Gap:  " << mBSNNGaps / mTotalNumInstance << std::endl;

		fcount++;
	}
	std::cout << std::endl;
	std::cout << std::endl;


	std::cout << "----- Large Results ------" << std::endl;
	//Large results
	double BSCostsLarge{ 0.0 };
	double NNCostsLarge{ 0.0 };
	double BSNNGapsLarge{ 0.0 };

	for (size_t i = 0; i < AllResultsLarge.size(); i++)
	{
		BSCostsLarge += AllResultsLarge[i][0];
		NNCostsLarge += AllResultsLarge[i][1];
		BSNNGapsLarge += AllResultsLarge[i][2];
	}
	size_t TotalNumInstanceLarge = AllResultsLarge.size();
	std::cout << "Avg BS Costs:  " << BSCostsLarge / TotalNumInstanceLarge << "  , Avg NN Costs:  " << NNCostsLarge / TotalNumInstanceLarge;
	std::cout << "  , Avg BS - NN Gap:  " << BSNNGapsLarge / TotalNumInstanceLarge << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

	//Shelf life results
	size_t mcountLarge = 0;
	for (size_t m : {3, 4, 5})
	{
		double mBSCostsLarge{ 0.0 };
		double mNNCostsLarge{ 0.0 };
		double mBSNNGapsLarge{ 0.0 };

		for (size_t i = 0; i < ShelfLifeResultsLarge[mcountLarge].size(); i++)
		{
			mBSCostsLarge += ShelfLifeResultsLarge[mcountLarge][i][0];
			mNNCostsLarge += ShelfLifeResultsLarge[mcountLarge][i][1];
			mBSNNGapsLarge += ShelfLifeResultsLarge[mcountLarge][i][2];
		}
		size_t mTotalNumInstanceLarge = ShelfLifeResultsLarge[mcountLarge].size();
		std::cout << "Product life:  " << m << "  Avg BS Costs:  " << mBSCostsLarge / mTotalNumInstanceLarge;
		std::cout << "  , Avg NN Costs:  " << mNNCostsLarge / mTotalNumInstanceLarge;
		std::cout << "  , Avg BS - NN Gap:  " << mBSNNGapsLarge / mTotalNumInstanceLarge << std::endl;

		mcountLarge++;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	//Lead time results
	size_t lcountLarge = 0;
	for (size_t l : {0, 1, 2})
	{
		double mBSCosts{ 0.0 };
		double mNNCosts{ 0.0 };
		double mBSNNGaps{ 0.0 };

		for (size_t i = 0; i < LeadTimeResultsLarge[lcountLarge].size(); i++)
		{
			mBSCosts += LeadTimeResultsLarge[lcountLarge][i][0];
			mNNCosts += LeadTimeResultsLarge[lcountLarge][i][1];
			mBSNNGaps += LeadTimeResultsLarge[lcountLarge][i][2];
		}
		size_t mTotalNumInstance = LeadTimeResultsLarge[lcountLarge].size();
		std::cout << "Lead time:  " << l << "  Avg BS Costs:  " << mBSCosts / mTotalNumInstance;
		std::cout << "  , Avg NN Costs:  " << mNNCosts / mTotalNumInstance;
		std::cout << "  , Avg BS - NN Gap:  " << mBSNNGaps / mTotalNumInstance << std::endl;

		lcountLarge++;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	//CVR results
	size_t ccountLarge = 0;
	for (double cvr : {1.0, 1.5, 2.0})
	{
		double mBSCosts{ 0.0 };
		double mNNCosts{ 0.0 };
		double mBSNNGaps{ 0.0 };

		for (size_t i = 0; i < CVRResultsLarge[ccountLarge].size(); i++)
		{
			mBSCosts += CVRResultsLarge[ccountLarge][i][0];
			mNNCosts += CVRResultsLarge[ccountLarge][i][1];
			mBSNNGaps += CVRResultsLarge[ccountLarge][i][2];
		}
		size_t mTotalNumInstance = CVRResultsLarge[ccountLarge].size();
		std::cout << "CVR:  " << cvr << "  Avg BS Costs:  " << mBSCosts / mTotalNumInstance;
		std::cout << "  , Avg NN Costs:  " << mNNCosts / mTotalNumInstance;
		std::cout << "  , Avg BS - NN Gap:  " << mBSNNGaps / mTotalNumInstance << std::endl;

		ccountLarge++;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	//f results
	size_t fcountLarge = 0;
	for (double f : {1.0, 0.5, 0.0})
	{
		double mBSCosts{ 0.0 };
		double mNNCosts{ 0.0 };
		double mBSNNGaps{ 0.0 };

		for (size_t i = 0; i < fResultsLarge[fcountLarge].size(); i++)
		{
			mBSCosts += fResultsLarge[fcountLarge][i][0];
			mNNCosts += fResultsLarge[fcountLarge][i][1];
			mBSNNGaps += fResultsLarge[fcountLarge][i][2];
		}
		size_t mTotalNumInstance = fResultsLarge[fcountLarge].size();
		std::cout << "Issuance Policy f:  " << f << "  Avg BS Costs:  " << mBSCosts / mTotalNumInstance;
		std::cout << "  , Avg NN Costs:  " << mNNCosts / mTotalNumInstance;
		std::cout << "  , Avg BS - NN Gap:  " << mBSNNGaps / mTotalNumInstance << std::endl;

		fcountLarge++;
	}
	std::cout << std::endl;
	std::cout << std::endl;


	std::cout << "----- Small Results ------" << std::endl;
	//Small results
	double Opt_gap { 0.0 };

	for (size_t i = 0; i < AllResultsSmall.size(); i++)
	{
		Opt_gap += AllResultsSmall[i];
	}
	size_t TotalNumInstanceSmall = AllResultsSmall.size();
	std::cout << "Avg Opt Gap:  " << Opt_gap / TotalNumInstanceSmall << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

	//Shelf life results
	size_t mcountSmall = 0;
	for (size_t m : {3, 4, 5})
	{
		double Opt_gap{ 0.0 };

		for (size_t i = 0; i < ShelfLifeResultsSmall[mcountSmall].size(); i++)
		{
			Opt_gap += ShelfLifeResultsSmall[mcountSmall][i];
		}
		size_t mTotalNumInstanceSmall = ShelfLifeResultsSmall[mcountSmall].size();
		std::cout << "Product life:  " << m << "  Avg Opt Gap:  " << Opt_gap / mTotalNumInstanceSmall << std::endl;

		mcountSmall++;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	//Lead time results
	size_t lcountSmall = 0;
	for (size_t l : {0, 1, 2})
	{
		double Opt_gap{ 0.0 };

		for (size_t i = 0; i < LeadTimeResultsSmall[lcountSmall].size(); i++)
		{
			Opt_gap += LeadTimeResultsSmall[lcountSmall][i];
		}
		size_t mTotalNumInstanceSmall = LeadTimeResultsSmall[lcountSmall].size();
		std::cout << "Lead time:  " << l << "  Avg Opt Gap:  " << Opt_gap / mTotalNumInstanceSmall << std::endl;

		lcountSmall++;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	//CVR results
	size_t ccountSmall = 0;
	for (double cvr : {1.0, 1.5, 2.0})
	{
		double Opt_gap{ 0.0 };

		for (size_t i = 0; i < CVRResultsSmall[ccountSmall].size(); i++)
		{
			Opt_gap += CVRResultsSmall[ccountSmall][i];
		}
		size_t mTotalNumInstanceSmall = CVRResultsSmall[ccountSmall].size();
		std::cout << "CVR:  " << cvr << "  Avg Opt Gap:  " << Opt_gap / mTotalNumInstanceSmall << std::endl;

		ccountSmall++;
	}
	std::cout << std::endl;
	std::cout << std::endl;

	//f results
	size_t fcountSmall = 0;
	for (double f : {1.0, 0.5, 0.0})
	{
		double Opt_gap{ 0.0 };

		for (size_t i = 0; i < fResultsSmall[fcountSmall].size(); i++)
		{
			Opt_gap += fResultsSmall[fcountSmall][i];
		}
		size_t mTotalNumInstanceSmall = fResultsSmall[fcountSmall].size();
		std::cout << "Issuance Policy f:  " << f << "  Avg Opt Gap:  " << Opt_gap / mTotalNumInstanceSmall << std::endl;

		fcountSmall++;
	}
	std::cout << std::endl;
	std::cout << std::endl;
}

int main() {

	TestPaperInstances();

	return 0;

}