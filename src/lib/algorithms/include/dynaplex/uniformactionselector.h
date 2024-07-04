#pragma once
#include "dynaplex/mdp.h"
#include "dynaplex/policy.h"
#include "dynaplex/system.h"
#include "dynaplex/vargroup.h"
#include "dynaplex/sample.h"

namespace DynaPlex::DCL {
	class UniformActionSelector {

	
	public:
		UniformActionSelector() = default;
		UniformActionSelector(int64_t rng_seed, DynaPlex::MDP&, DynaPlex::Policy&, bool SimulateOnlyPromisingActions, int64_t Num_Promising_Actions);

		void SetAction(DynaPlex::Trajectory& traj, DynaPlex::NN::Sample& sample, int64_t seed, const int64_t H, const int64_t M) const;


	

	private:
		int64_t  rng_seed;
		int64_t Num_Promising_Actions;
		bool SimulateOnlyPromisingActions;
		DynaPlex::Policy policy;
		DynaPlex::MDP mdp;

	};
}//namespace DynaPlex::Utilities