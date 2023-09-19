#pragma once
#include <cstdint>
#include <vector>
#include "vargroup.h"
#include "statecategory.h"
#include "rngprovider.h"
#include "state.h"
#include "system.h"

namespace DynaPlex {

	struct Trajectory {
		/// used by MDPAdapter when IncorporateActions is called without argument.
	    int64_t NextAction;
		/**
		  * The Category associated with the current state; kept up to date when making calls to MDP->func(Trajectories, ...)
		  * Do not manually change this. 
	   	  */
		StateCategory Category;
		/**
		  * counts the number of events since the last initiation/reset. 
		  * Automatically kept up-to-date with calls to MDP->func(Trajectories, ...). 
		  * Do not manually change this.
		  */
		int64_t EventCount;
		/**
	      * Effective discount factor based on the number of events since initiation / last reset. 
		  * Multiplied by the DiscountFactor before each event: Automatically kept up-to-date with calls to MDP->func(Trajectories, ...).
          * Do not manually change this.
          */
		double EffectiveDiscountFactor;

		/**
	      * Cumulative return since initiation/last reset. 
	      * Automatically kept up-to-date with calls to MDP->func(Trajectories, ...).
	      * Do not manually change this.
	      */
		double CumulativeReturn;
	private:
		/// (type-erased) state of the system.
		DynaPlex::dp_State state;
	public:
		DynaPlex::dp_State& GetState()
		{
			if (state)
			{
				return state;
			}
			else
				throw DynaPlex::Error("Trajectory: Attempting to get state that has not been initialized. ");
		}
		/// moves the state into the trajectory, and resets CumulativeReturn, EffectiveDiscountFactor, and EventCount. 
		void Reset(DynaPlex::dp_State&& State);
		
		/// re-initiates EventCount, EffectiveDiscountFactor, CumulativeReturn.  
		void Reset();
		/// provider of random sequences for use in MDP. May be reset by calling Trajectory::Reset. 
		DynaPlex::RNGProvider RNGProvider;
		/**
		 * Convenience member that may be used to store the (index of) an object in a container that contains more information
		 * on this trajectory, like initial information or functions to call when the trajectory completes.
		 */
		int64_t ExternalIndex;	
	


		/// resets the RNGProvider, ensuring consistent random numbers when comparing things.
		void SeedRNGProvider(const DynaPlex::System& system, bool eval, int64_t experiment_number=0, uint32_t thread_number=0);

		/**
		 * Creates a trajectory that supports NumEventRNGs, without initial state.
		 * external_index is some context_defined item that is not muted by MDPAdapter. 
		 */
		Trajectory(int64_t NumEventRNGs, int64_t externalIndex = 0);


		/// converts Trajectory to VarGroup
		VarGroup ToVarGroup() const;

	
	};
}