#include "dynaplex/policycomparer.h"
#include "dynaplex/trajectory.h"
#include "dynaplex/parallel_execute.h"
#include "dynaplex/policycomparison.h"
namespace DynaPlex::Utilities {

	void PolicyComparer::ComputeReturns(std::span<std::vector<double>>& ReturnPerTrajectory,const DynaPlex::Policy& policy, int64_t offset) const
	{
		std::vector<DynaPlex::Trajectory> trajectories{};
		trajectories.reserve(ReturnPerTrajectory.size());

		for (int64_t experiment_number = 0; experiment_number < ReturnPerTrajectory.size(); experiment_number++)
		{
			trajectories.emplace_back(experiment_number + offset);
			trajectories.back().RNGProvider.SeedEventStreams(true, rng_seed, experiment_number + offset);
		}

		//Initiate each trajectory with a random state. 
		mdp->InitiateState(trajectories);		

		if (mdp->IsInfiniteHorizon())
		{
			//Only do a warm-up for the undiscounted case. 
			//For discounted case, we are trying to assess the average cumulative return from Initial State
			//during a trajectory of length periods_per_trajectory. 

			//For undiscounted case, we try to assess average cost per period. 
			if (mdp->DiscountFactor() == 1.0)
			{
				Evolve(policy, trajectories, warmup_periods);
				for (size_t i = 0; i < ReturnPerTrajectory.size(); i++)
					ReturnPerTrajectory[i].front() = trajectories[i].CumulativeReturn;
				if (number_of_statistics > 0)
					mdp->InitiateStateVariables(trajectories);
				CheckTrajectoriesInfiniteHorizon(trajectories, warmup_periods);
			}
			else
				if (warmup_periods != 0)
					throw DynaPlex::Error("PolicyComparer: Error in logic - warmup_periods should be zero for discounted cost logic.");

			Evolve(policy, trajectories, warmup_periods + periods_per_trajectory);
			CheckTrajectoriesInfiniteHorizon(trajectories, warmup_periods + periods_per_trajectory);
			for (size_t i = 0; i < ReturnPerTrajectory.size(); i++)
			{
				ReturnPerTrajectory[i].front() = trajectories[i].CumulativeReturn - ReturnPerTrajectory[i].front();
			}
			if (mdp->DiscountFactor() == 1)
			{
				for (auto& returnVal : ReturnPerTrajectory)
					returnVal.front() /= periods_per_trajectory;
			}
		}
		else
		{//finite horizon:
			Evolve(policy, trajectories, max_periods_until_error);
			CheckTrajectoriesFiniteHorizon(trajectories);
			for (size_t i = 0; i < ReturnPerTrajectory.size(); i++)
			{
				ReturnPerTrajectory[i].front() = trajectories[i].CumulativeReturn;
			}
		}
		if (number_of_statistics > 0)
		{
			for (size_t i = 0; i < ReturnPerTrajectory.size(); i++)
			{
				std::vector<double> stats = mdp->ReturnUsefulStatistics(trajectories[i].GetState());
				if (!stats.size() == number_of_statistics)
					throw DynaPlex::Error("PolicyComparer: Error in logic - size of the useful statistics should be equal to number_of_statistics.");
				for (size_t j = 1; j <= number_of_statistics; j++) {
					ReturnPerTrajectory[i][j] = stats[j - 1];
				}
			}			
		}
	}

	PolicyComparer::PolicyComparer(const DynaPlex::System& system, DynaPlex::MDP mdp, const VarGroup& config)
		: system{ system }, mdp{ mdp }
	{
		if (!mdp)
		{
			throw DynaPlex::Error("PolicyComparer: parameter MDP should not be null");
		}
		// Default values based on MDP horizon
		if (mdp->IsInfiniteHorizon()) {
			config.GetOrDefault("number_of_trajectories", number_of_trajectories, 4096);
			config.GetOrDefault("periods_per_trajectory", periods_per_trajectory, 1024);

			if (mdp->DiscountFactor() == 1.0)
			{
				config.GetOrDefault("warmup_periods", warmup_periods, 128);
			}
			else
			{
				warmup_periods = 0;
			}
			max_periods_until_error = 0; // Unused for infinite horizon. 
		}
		else {
			config.GetOrDefault("number_of_trajectories", number_of_trajectories, 16384);
			config.GetOrDefault("max_periods_until_error", max_periods_until_error, 16384);
			periods_per_trajectory = 0;  // Unused for finite horizon MDP
			warmup_periods = 0; //also unused. 
		}
		config.GetOrDefault("rng_seed", rng_seed, 13021984);
		config.GetOrDefault("number_of_statistics", number_of_statistics, 0);
		config.GetOrDefault("avoidable_cost", avoidable_cost, false);
		config.GetOrDefault("print_standard_error", print_standard_error, false);
		if (rng_seed < 0)
			throw DynaPlex::Error("PolicyComparer :: Invalid rng_seed - should be non-negative");
	}

	void PolicyComparer::CheckTrajectoriesInfiniteHorizon(std::span<DynaPlex::Trajectory> trajectories, int64_t cumulative_periods) const {
		for (auto& traj : trajectories)
		{
			if (traj.Category.IsFinal())
			{
				throw DynaPlex::Error("PolicyComparer: mdp " + mdp->TypeIdentifier() + " has infinite horizon but returns categories that are IsFinal.");
			}
			if (traj.PeriodCount != cumulative_periods)
			{
				throw DynaPlex::Error("PolicyComparer: Error in logic for mdp: " + mdp->TypeIdentifier()+". Please contact developers.");
			}
		}
	}
	void PolicyComparer::CheckTrajectoriesFiniteHorizon(std::span<DynaPlex::Trajectory> trajectories) const {
		for (auto& traj : trajectories)
		{
			if (!traj.Category.IsFinal())
			{
				throw DynaPlex::Error("PolicyComparer: mdp " + mdp->TypeIdentifier() + " has finite horizon but does not reach state with IsFinal after max_periods_until_error=" + std::to_string(max_periods_until_error) + " periods. ");
			}
		}
	}


	template<bool SkipTrivialActions>
	void PolicyComparer::Evolve(const DynaPlex::Policy& policy, std::span<DynaPlex::Trajectory> span, int64_t max_periods) const
	{
		while (true)
		{
			bool all_require_action;
			if constexpr (SkipTrivialActions)
			{
				all_require_action = mdp->IncorporateUntilAction(span, max_periods);
			}
			else
			{
				all_require_action = mdp->IncorporateUntilNonTrivialAction(span, max_periods);
			}
			if (!all_require_action)
			{
				//This "sorts" the trajectories, such that the trajectories that are IsAwaitAction are at the front.
				// Note that mdp->IncorporateAction only accepts set of adjacent trajectories that are all AwaitAction. 
				std::span<DynaPlex::Trajectory>::iterator new_partition_point = std::partition(span.begin(), span.end(),
					[](const DynaPlex::Trajectory& traj) {return traj.Category.IsAwaitAction(); }
				);
				//new_partition_point is the first element which does not require an action. 
				//Make the span refer to a set of trajectories each awaiting an action:
				span = std::span<DynaPlex::Trajectory>(span.begin(), new_partition_point);
			}
			//This means all trajectories are at period warmup_periods or final. 
			if (span.size() == 0)
				break;
			mdp->IncorporateAction(span, policy);
		}
	}

	DynaPlex::VarGroup PolicyComparer::Assess(DynaPlex::Policy policy) const {
		std::vector<DynaPlex::Policy> polVec{};
		polVec.reserve(1);
		polVec.push_back(policy);
		return Compare(polVec)[0];		
	}

	std::vector<VarGroup> PolicyComparer::Compare(DynaPlex::Policy first, DynaPlex::Policy second, int64_t index_of_benchmark, bool compute_gap, bool rewardProblem) const {
		std::vector<DynaPlex::Policy> polVec{};
		polVec.reserve(2);
		polVec.push_back(first);
		polVec.push_back(second);
		return Compare(polVec, index_of_benchmark, compute_gap, rewardProblem);
	}

	std::vector<VarGroup> PolicyComparer::Compare(std::vector<DynaPlex::Policy> policies, int64_t index_of_benchmark, bool compute_gap, bool rewardProblem) const {
		std::vector<std::vector<double>> ReturnValues{};
		ReturnValues.reserve(policies.size());
		std::vector<std::vector<std::vector<double>>> ReturnStatistics{};
		ReturnStatistics.reserve(policies.size());
		int64_t minusone = -1, size = policies.size();
		if (!(index_of_benchmark >= minusone && index_of_benchmark < size))
		{
			throw DynaPlex::Error("PolicyComparer: invalid value for index_of_benchmark; should be -1 or an index corresponding to a policy. Actual value: " + std::to_string(index_of_benchmark));
		}
		
		for (int i=0;i<policies.size();i++)
		{
			auto& policy = policies[i];
			if (!policy) {
				throw DynaPlex::Error("PolicyComparer: policy should not be null");
			}
			std::vector<std::vector<double>> nestedVec{};
			nestedVec.reserve(number_of_trajectories);
			for (int j = 0; j < number_of_trajectories; j++)
			{
				nestedVec.push_back(std::vector<double>(number_of_statistics + 1, 0.0));
			}
			DynaPlex::Parallel::parallel_compute<std::vector<double>>(nestedVec, [this, &policy](std::span<std::vector<double>> span, int64_t start) {
				this->ComputeReturns(span, policy, start);
				}, system.HardwareThreads());	

			std::vector<double> returnCostValues;
			returnCostValues.reserve(number_of_trajectories);
			std::vector<std::vector<double>> returnStatisticValues;
			if (number_of_statistics > 0) {
				returnStatisticValues.reserve(number_of_statistics);
				for (int j = 0; j < number_of_statistics; j++)
				{
					returnStatisticValues.push_back(std::vector<double>(number_of_trajectories, 0.0));
				}
			}
			for (int k = 0; k < number_of_trajectories; k++) {
				auto returnVal = nestedVec[k];
				returnCostValues.push_back(returnVal.front());
				if (number_of_statistics > 0) {
					for (int j = 1; j <= number_of_statistics; j++)
						returnStatisticValues[j - 1][k] = returnVal[j];
				}
			}
			ReturnValues.push_back(returnCostValues);
			if (number_of_statistics > 0) 
				ReturnStatistics.push_back(returnStatisticValues);
		}

		bool MdpAverageCost = (mdp->IsInfiniteHorizon() && mdp->DiscountFactor() == 1.0) ? true : false;
		DynaPlex::PolicyComparison comparison{ ReturnValues };
		std::vector<DynaPlex::VarGroup> varGroups;
		varGroups.reserve(policies.size());
		for (size_t i = 0; i < policies.size(); i++)
		{
			auto& policy = policies[i];
			DynaPlex::VarGroup forPolicy{};
			forPolicy.Add("policy", policy->GetConfig());
			forPolicy.Add("mean", comparison.mean(i));
			if (number_of_trajectories > 1 && print_standard_error)
				forPolicy.Add("st_error", comparison.standardError(i));
			if (index_of_benchmark > minusone && i != index_of_benchmark)
			{
				forPolicy.Add("mean_difference", comparison.mean(i, index_of_benchmark));
				if (number_of_trajectories > 1 && print_standard_error)
					forPolicy.Add("st_error_difference", comparison.standardError(i, index_of_benchmark));
				if (compute_gap)
				{
					if (!rewardProblem)
						forPolicy.Add("mean_gap", (comparison.mean(i) - comparison.mean(index_of_benchmark)) / comparison.mean(index_of_benchmark) * 100);
					else
						forPolicy.Add("mean_gap", (comparison.mean(index_of_benchmark) - comparison.mean(i)) / comparison.mean(index_of_benchmark) * 100);
				}
			}
			if (number_of_statistics > 0)
			{
				DynaPlex::PolicyComparison secondary_comparison{ ReturnStatistics[i] };
				for (int64_t j = 0; j < number_of_statistics; j++)
				{
					forPolicy.Add("mean_stat_" + std::to_string(static_cast<int64_t>(j + 1)), secondary_comparison.mean(j));
					if (number_of_trajectories > 1 && print_standard_error)
						forPolicy.Add("st_error_stat_" + std::to_string(static_cast<int64_t>(j + 1)), secondary_comparison.standardError(j));
				}
				if (avoidable_cost) {
					if (!(index_of_benchmark > minusone && index_of_benchmark < size))
					{
						throw DynaPlex::Error("PolicyComparer: invalid value for index_of_benchmark for avoidable cost calculation.");
					}

					std::vector<std::vector<double>> avoidable_costs{};
					avoidable_costs.reserve(2);
					std::vector<double> policy_cost{};
					policy_cost.reserve(number_of_trajectories);
					std::vector<double> benchmark_cost{};
					benchmark_cost.reserve(number_of_trajectories);
					if (MdpAverageCost) {
						for (int64_t j = 0; j < number_of_trajectories; j++) {
							double val = ReturnStatistics[index_of_benchmark].back()[j];
							policy_cost.push_back((ReturnValues[i][j] * periods_per_trajectory - val) / periods_per_trajectory);
							benchmark_cost.push_back((ReturnValues[index_of_benchmark][j] * periods_per_trajectory - val) / periods_per_trajectory);
						}
					}
					else {
						for (int64_t j = 0; j < number_of_trajectories; j++) {
							double val = ReturnStatistics[index_of_benchmark].back()[j];
							policy_cost.push_back(ReturnValues[i][j] - val);
							benchmark_cost.push_back(ReturnValues[index_of_benchmark][j] - val);
						}
					}
					avoidable_costs.push_back(policy_cost);
					avoidable_costs.push_back(benchmark_cost);

					DynaPlex::PolicyComparison avoidable_cost_comparison{ avoidable_costs };
					forPolicy.Add("av_mean", avoidable_cost_comparison.mean(0));
					if (number_of_trajectories > 1 && print_standard_error)
						forPolicy.Add("av_st_error", avoidable_cost_comparison.standardError(0));
					forPolicy.Add("av_mean_difference", avoidable_cost_comparison.mean(0, 1));
					if (number_of_trajectories > 1 && print_standard_error)
						forPolicy.Add("av_st_error_difference", avoidable_cost_comparison.standardError(0, 1));
					if (compute_gap)
					{
						if (!rewardProblem)
							forPolicy.Add("av_mean_gap", (avoidable_cost_comparison.mean(0) - avoidable_cost_comparison.mean(1)) / avoidable_cost_comparison.mean(1) * 100);
						else
							forPolicy.Add("av_mean_gap", (avoidable_cost_comparison.mean(1) - avoidable_cost_comparison.mean(0)) / avoidable_cost_comparison.mean(1) * 100);
					}
				}
			}
			//if (i == index_of_benchmark)
			//{
			//	forPolicy.Add("benchmark", "yes");
			//}
			varGroups.push_back(forPolicy);
		}

		return varGroups;
	}
}  // namespace DynaPlex::Utilities
