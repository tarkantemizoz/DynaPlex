#pragma once
#include "dynaplex/mdp.h"
#include "dynaplex/policy.h"
#include "dynaplex/system.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Algorithms {
	class ExactSolver
	{
	public:
		/**
		 * @brief Solver for computing exact policy costs and exact optimal costs. <Explain what is and is not supported>.
		 * @param system object
		 * @param mdp model
		 * @param config file (optional), that may provide:
		*/
		ExactSolver(const DynaPlex::System& system, DynaPlex::MDP mdp, const DynaPlex::VarGroup& config = VarGroup{});

		/**
		 * @brief Computes exact return (costs/reward depending on mdp) for the policy. Computes exact optimal costs if no policy is provided.
		 * @param Policy for the MDP
		 * @return For finite-horizon MDP and discounted MDP: Expected return when starting in the initial state.
		*/
		double ComputeCosts(DynaPlex::Policy policy = nullptr);


		/**
		 * @brief returns the optimal policy. Will reuse the optimal policy from ComputeCosts if available, otherwise initiated the exact solution process.
		 * @return the optimal policy for the mdp, computed using a hybrid policy iteration/value iteration algorithm.
		 */
		DynaPlex::Policy GetOptimalPolicy();
	private:
		friend class ExactPolicy;
		class Impl;
		std::shared_ptr<Impl> pImpl;
	};
}