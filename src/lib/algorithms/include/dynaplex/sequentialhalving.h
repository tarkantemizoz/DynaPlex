#pragma once
#include "dynaplex/mdp.h"
#include "dynaplex/policy.h"
#include "dynaplex/system.h"
#include "dynaplex/vargroup.h"
#include "dynaplex/sample.h"

namespace DynaPlex::DCL {
	/**
	* Sequential Halving algorithm. 
	* A state-of-the-art bandit algorithm for selecting the best alternative out of others.
	* Automatically kept up-to-date with calls to MDP->func(Trajectories, ...).
	* See the original paper: https://proceedings.mlr.press/v28/karnin13.pdf
	* See DCL paper for its implementation: https://arxiv.org/pdf/2011.15122.pdf
	*/
	class SequentialHalving {


	public:
		SequentialHalving() = default;
		SequentialHalving(int64_t rng_seed, DynaPlex::MDP&, DynaPlex::Policy&, bool SimulateOnlyPromisingActions, int64_t Num_Promising_Actions);

		void SetAction(DynaPlex::Trajectory& traj, DynaPlex::NN::Sample& sample, int64_t seed, const int64_t H, const int64_t M) const;




	private:
		int64_t rng_seed;
		int64_t Num_Promising_Actions;
		bool SimulateOnlyPromisingActions;
		DynaPlex::Policy policy;
		DynaPlex::MDP mdp;

	};
}//namespace DynaPlex::Utilities