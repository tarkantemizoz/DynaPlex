#include "dynaplex/exactsolver.h"
#include "dynaplex/error.h"
#include <forward_list>

namespace DynaPlex::Algorithms {
	class ExactSolver::Impl {
	public:

		static constexpr size_t float_size = sizeof(float);
		System system;
		DynaPlex::MDP mdp;
		bool silent;
		bool exact_policy_computed{ false };
		bool exact_policy_exported{ false };
		//for numerical stability of hybrid iteration algorithm when discountfactor = 1, 
		//we need to add self-transitions to avoid periodicity. 
		static constexpr double self_transition_prob = 0.02;
		Impl(const System& sys, DynaPlex::MDP mdp, const DynaPlex::VarGroup& conf)
			: system(sys), mdp(mdp), hasher{}, statemap{}, action_states{} {

			conf.GetOrDefault("silent", silent, false);

			if (!mdp->ProvidesEventProbs()) {
				throw DynaPlex::Error("ExactSolver: MDP does not provide event probabilities. Note that exact algorithms need event probabilities.");
			}
			if (!mdp->ProvidesFlatFeatures()) {
				throw DynaPlex::Error("ExactSolver: MDP does not provide flat features. Note that features are used to determine state equality.");
			}

			if (mdp->DiscountFactor() == 1.0)
			{
				if (!silent) {
					if (mdp->IsInfiniteHorizon())
					{
						std::string Message = "ExactSolver: DiscountFactor==1.0 & infinite horizon model: MDP should be at least weakly communicating for convergence, perhaps stronger conditions are needed.";
						system << Message << std::endl;
					}
					else
					{
						std::string Message = "ExactSolver: discount factor is 1.0 & finite-horizon MDP. Assumption is made that eventually a final state is always reached. ";
						system << Message << std::endl;
					}
				}
			}

			conf.GetOrDefault("max_states", max_states, 1048576);
			feats_holder.resize(mdp->NumFlatFeatures(), 0.0f);
			feats_holder2.resize(mdp->NumFlatFeatures(), 0.0f);

		}
		std::hash<float> hasher;
		//helper object that will be reused, to avoid very frequent memory allocations;
		std::vector<float> feats_holder, feats_holder2;
		//helper object that will be reused, to avoid very frequent memory allocations;
		std::vector<std::tuple<double, DynaPlex::dp_State>> transitions_holder;
		//helper object that will be reused, to avoid very frequent memory allocations;
		std::vector<DynaPlex::Trajectory> trajectories;

		int64_t max_states;


		bool StatesAreEqual(const DynaPlex::dp_State& state, const DynaPlex::dp_State& state2)
		{
			//could have relied on mdp->StatesAreEqual here, but that would require humans to implement
			//state equality, which sometimes causes issues. Since we need a hash anyhow, and thus some features,
			//this seems easier. 
			mdp->GetFlatFeatures(state, feats_holder);
			mdp->GetFlatFeatures(state2, feats_holder2);
			//states are considered equal if their features are equal. 
			return feats_holder == feats_holder2;
		}

		size_t GetHashThreadSafe(const DynaPlex::dp_State& state) {
			std::vector<float> feats_holder_ts;
			feats_holder_ts.resize(mdp->NumFlatFeatures(), 0.0f);
			mdp->GetFlatFeatures(state, feats_holder_ts);
			size_t hash_key = 0;
			for (const float& f : feats_holder_ts) {
				hash_key ^= hasher(f) + 0x9e3779b9 + (hash_key << 6) + (hash_key >> 2);
			}
			return hash_key;
		}

		size_t GetHash(const DynaPlex::dp_State& state) {
			mdp->GetFlatFeatures(state, feats_holder);
			size_t hash_key = 0;
			for (const float& f : feats_holder) {
				hash_key ^= hasher(f) + 0x9e3779b9 + (hash_key << 6) + (hash_key >> 2);
			}
			return hash_key;
		}

		struct LookupValue {
			size_t state_index;
			double value;
			LookupValue(size_t state_index) : state_index{ state_index }, value{ 0.0 } {

			}
		};

		struct Transition {
			double probability;
			LookupValue* values;
		};

		struct StateStorage {
			DynaPlex::dp_State state;
			int64_t current_action;
			double new_value;
			//expected costs per period
			double costs_until_transition;
			std::vector<Transition> transitions;
			StateStorage(DynaPlex::dp_State&& state) :
				state{ std::move(state) }, current_action{ 0 }, new_value{ 0.0 }
			{

			}
		};

		//key data structures:
		std::unordered_map<size_t, std::forward_list<LookupValue>> statemap;
		std::vector<StateStorage> action_states;
		//to keep count of any hash collisions 
		size_t hash_collisions;

		LookupValue& GetStateWithHash(const DynaPlex::dp_State& state, size_t hash) {
			auto iter = statemap.find(hash);
			if (iter == statemap.end()) {
				auto cat = mdp->GetStateCategory(state);
				if (!cat.IsAwaitAction())
					throw DynaPlex::Error("Attempting to get state value for non-action state.");
				throw DynaPlex::Error("Hash key not found in statemap.");
			}
			auto& list = iter->second;
			auto it = list.begin();
			auto next_it = it;
			// If there is only one element, return it without comparison.
			if (++next_it == list.end())
				return *it;

			// If there are multiple elements, find the correct one by comparing state equality.
			for (auto& lookup : list) {
				if (StatesAreEqual(state, action_states[lookup.state_index].state)) {
					return lookup;
				}
			}
			throw DynaPlex::Error("State expected to be found but was not. ");
		}

		LookupValue& GetStateValue(const DynaPlex::dp_State& state) {
			auto hash = GetHash(state);
			return GetStateWithHash(state, hash);
		}
		LookupValue& GetStateValueTS(const DynaPlex::dp_State& state) {
			auto hash = GetHashThreadSafe(state);
			return GetStateWithHash(state, hash);
		}

		//Checks if state is allready added to list, and adds to list if not present. 
		void AddState(DynaPlex::dp_State& state) {
			auto hash = GetHash(state);
			bool found = false;
			if (statemap.contains(hash)) {
				for (const auto& idx_and_value : statemap[hash]) {
					if (StatesAreEqual(state, action_states[idx_and_value.state_index].state)) {
						found = true;
						break;
					}
				}
				//if the hash was found but the state was not, then we encountered a collision. 
				if (!found)
					hash_collisions++;
			}
			if (!found) {
				if (action_states.size() >= max_states)
				{
					std::string message = "ExactSolver: Number of action states in mdp exceeds option max_states (=";
					message += std::to_string(max_states); message += "). It may not be feasible to solve this MDP exactly. Consider adapting the max_states option.";
					throw DynaPlex::Error(message);
				}
				statemap[hash].emplace_front(action_states.size());
				action_states.push_back(std::move(state));
			}
		}

		///Processes this state - Adds to list if action state (for later expansion), or expand immediately
		///if event state. 
		void ProcessState(DynaPlex::dp_State& state, size_t depth = 0)
		{
			auto category = mdp->GetStateCategory(state);
			if (category.IsAwaitAction())
				//added to list, to be expanded later. 
				AddState(state);
			else if (category.IsAwaitEvent()) {  //expand this event state immediately. 
				if (depth > 6)
					throw DynaPlex::Error("ExactSolver: Failed attempt to map states for MDP that has many subsequent events. To succeed, ensure dummy actions after every event.");
				transitions_holder.clear();
				mdp->GetAllEventTransitions(state, transitions_holder);
				for (auto& [prob, tr_state] : transitions_holder)
				{
					auto tr_category = mdp->GetStateCategory(tr_state);
					if (tr_category.IsAwaitEvent())
						throw DynaPlex::Error("ExactSolver: Event states following immediately after event states in MDP; this is currently not supported. MDP is expected to alternate between events and actions, and may transition to final state.");
					ProcessState(tr_state, depth + 1);
				}
			}
		}

		/// Expands this action state, and processes all child states.
		void ExpandActionState(DynaPlex::dp_State& state) {
			auto allowedactions = mdp->AllowedActions(state);
			trajectories.resize(allowedactions.size());
			mdp->InitiateState(trajectories, state);
			size_t iter = 0;
			for (auto action : allowedactions)
				trajectories[iter++].NextAction = action;
			mdp->IncorporateAction(trajectories);
			for (DynaPlex::Trajectory& traj : trajectories)
			{
				if (traj.Category.IsAwaitAction())
					throw DynaPlex::Error("ExactSolver: Action states following immediately after action states in MDP; this is currently not supported. MDP is expected to alternate between events and actions, and may transition to final state.");
				if (traj.Category.Index() != 0)
					throw DynaPlex::Error("ExactSolver: Event states must have index 0, i.e. correspond to a time step, in current version of ExactSolver. ");

				ProcessState(traj.GetState());
			}
		}

		//This sets actions in the action_states following the policy. 
		void SetActions(DynaPlex::Policy policy)
		{
			Policy pol;
			if (policy)
				pol = policy;
			else
				pol = mdp->GetPolicy("greedy");
			DynaPlex::Trajectory traj{};
			traj.RNGProvider.SeedEventStreams(false);
			for (StateStorage& storage : action_states)
			{
				mdp->InitiateState({ &traj,1 }, storage.state);
				pol->SetAction({ &traj,1 });
				storage.current_action = traj.NextAction;
			}
		}
		//This populates/determines the transitions and costs for the actions currently set 
		//in the action_states vector. 
		void DetermineTransitions()
		{
			for (StateStorage& storage : action_states)
			{
				trajectories.clear();
				//create a new trajectory
				trajectories.emplace_back();
				mdp->InitiateState(trajectories, storage.state);
				trajectories[0].NextAction = storage.current_action;
				mdp->IncorporateAction(trajectories);
				if (trajectories[0].Category.IsAwaitEvent() && trajectories[0].Category.Index() == 0)
				{
					transitions_holder.clear();
					double expected_costs =
						mdp->GetAllEventTransitions(trajectories[0].GetState(), transitions_holder);
					storage.transitions.reserve(transitions_holder.size());
					storage.transitions.clear();
					for (auto& [prob, state] : transitions_holder)
					{
						auto cat = mdp->GetStateCategory(state);
						if (cat.IsAwaitAction())
						{
							auto& value = GetStateValue(state);
							storage.transitions.emplace_back(prob, &value);
						}
						else
						{
							if (cat.IsAwaitEvent())
								throw DynaPlex::Error("Events after events - this is not currently supported.");
						}
					}
					storage.costs_until_transition = mdp->DiscountFactor() * expected_costs + trajectories[0].CumulativeReturn;
				}
				else
				{
					if (trajectories[0].Category.IsAwaitAction())
						throw DynaPlex::Error("ExactSolver: Action states following immediately after action states in MDP; this is currently not supported. MDP is expected to alternate between events and actions, and may transition to final state.");
					if (trajectories[0].Category.Index() != 0)
						throw DynaPlex::Error("ExactSolver: Event states must have index 0, i.e. correspond to a time step, in current version of ExactSolver. ");

				}

			}
		}

		///This populates the key data structures action_states and statemap
		void CreateStateMap()
		{
			hash_collisions = 0;
			auto state = mdp->GetInitialState();
			ProcessState(state);
			size_t expanded_action_states = 0;
			while (expanded_action_states < action_states.size())
			{
				ExpandActionState(action_states[expanded_action_states].state);
				expanded_action_states++;
			}
			if (!silent) {
				system << "Created complete state list consisting of " << action_states.size() << " states." << std::endl;
				if (hash_collisions > 0)
					system << "There are " << hash_collisions << " collided hashes. " << std::endl;
				if (hash_collisions > 0.05 * action_states.size())
					system << "Such a high number of hash collisions is unexpected.";
				system << "Initiating hybrid value/policy iteration." << std::endl;
			}
		}
		double maxChange;
		double currentCost;

		void IterateValues() {

			for (auto& [key, list] : statemap)
				for (auto& lookupValue : list)
					lookupValue.value = action_states[lookupValue.state_index].new_value;


			for (auto& stateStorage : action_states) {
				stateStorage.new_value = 0;
				for (auto& transition : stateStorage.transitions)
				{
					stateStorage.new_value += transition.probability * (*transition.values).value;
				}
				stateStorage.new_value *= mdp->DiscountFactor();
				stateStorage.new_value += stateStorage.costs_until_transition;
			}
			if (mdp->IsInfiniteHorizon() && mdp->DiscountFactor() == 1.0)
			{	//This is a tricky case. 
				//to deal with/remove any periodicity that may be present in the model, 
				// we change to model to add cost-less transitions from states to themselves with
				//probability self_transition_prob. This increases probability of convergence.
				//note that such transitions do not affect the optimal policy (i.e. the system simply does "nothing" for a time-step).
				//however, the effective cost rate per period is reduced - it gets multiplied with (1.0 - self_transition_prob), which we
				//correct for when computing costs. 
				double no_self_transition_prob = 1.0 - self_transition_prob;
				for (auto& [key, list] : statemap) {
					for (auto& lookupValue : list)
					{
						action_states[lookupValue.state_index].new_value *= no_self_transition_prob;
						action_states[lookupValue.state_index].new_value += self_transition_prob * lookupValue.value;
					}
				}
			}
		}

		void CheckConvergence() {
			double deltaMax = -std::numeric_limits<double>::infinity();
			double deltaMin = std::numeric_limits<double>::infinity();
			double lowestValue = std::numeric_limits<double>::infinity();
			for (auto& [key, list] : statemap)
			{
				for (auto& lookupValue : list)
				{
					double delta{ action_states[lookupValue.state_index].new_value - lookupValue.value };
					deltaMax = std::max(deltaMax, delta);
					deltaMin = std::min(deltaMin, delta);
					lowestValue = std::min(lowestValue, action_states[lookupValue.state_index].new_value);
				}
			}

			for (auto& [key, list] : statemap)
			{
				for (auto& lookupValue : list)
				{
					action_states[lookupValue.state_index].new_value -= lowestValue;
					lookupValue.value -= lowestValue;
				}
			}
			if (mdp->IsInfiniteHorizon() && mdp->DiscountFactor() == 1.0)
			{
				maxChange = (deltaMax - deltaMin) / 2.0 / (1.0 - self_transition_prob);
				currentCost = (deltaMax + deltaMin) / 2.0 / (1.0 - self_transition_prob);
			}
			else
			{
				auto& LookupValue = GetStateValue(mdp->GetInitialState());
				currentCost = action_states[LookupValue.state_index].new_value;
				maxChange = std::max(deltaMax, -deltaMin);
			}
			if (!silent) {
				system << "Current return: " << currentCost << " Convergence:" << maxChange << std::endl;
			}

		}

		void UpdateActionsForValues() {

			double objective = mdp->Objective();
			for (auto& stateStorage : action_states) {
				auto allowed_actions = mdp->AllowedActions(stateStorage.state);
				double best_action_return = -std::numeric_limits<double>::infinity();
				stateStorage.current_action = std::numeric_limits<int64_t>::max();
				for (auto current_action : allowed_actions)
				{
					trajectories.clear();
					trajectories.emplace_back();
					mdp->InitiateState(trajectories, stateStorage.state);
					trajectories[0].NextAction = current_action;
					mdp->IncorporateAction(trajectories);

					if (trajectories[0].Category.IsAwaitEvent() && trajectories[0].Category.Index() == 0)
					{
						transitions_holder.clear();
						double direct_return =
							mdp->GetAllEventTransitions(trajectories[0].GetState(), transitions_holder);
						double expected_future_return = 0.0;
						for (auto& [prob, state] : transitions_holder)
						{
							auto cat = mdp->GetStateCategory(state);
							if (cat.IsAwaitAction())
							{
								auto& LookupValue = GetStateValue(state);
								expected_future_return += prob * LookupValue.value;
								if (!StatesAreEqual(action_states[LookupValue.state_index].state, state))
									throw DynaPlex::Error("issue with state retrieval while updating actions.");
							}
							else
							{
								if (cat.IsAwaitEvent())
									throw DynaPlex::Error("Events after events - this is not currently supported.");
							}
						}
						auto total_return = trajectories[0].CumulativeReturn + (mdp->DiscountFactor() * (expected_future_return + direct_return));
						total_return *= objective;
						if (total_return > best_action_return)
						{
							best_action_return = total_return;
							stateStorage.current_action = current_action;
						}
					}
					else
					{
						if (trajectories[0].Category.IsAwaitAction())
							throw DynaPlex::Error("ExactSolver: Action states following immediately after action states in MDP; this is currently not supported. MDP is expected to alternate between events and actions, and may transition to final state.");
						if (trajectories[0].Category.Index() != 0)
							throw DynaPlex::Error("ExactSolver: Event states must have index 0, i.e. correspond to a time step, in current version of ExactSolver. ");

					}
				}
				if (stateStorage.current_action == std::numeric_limits<int64_t>::max())
					throw DynaPlex::Error("current_action not updated in call to UpdateActionsForValues.");
				if (!mdp->IsAllowedAction(stateStorage.state, stateStorage.current_action))
					throw DynaPlex::Error("current_action is not legal after call to UpdateActionsForValues.");
			}

		}

		double ComputeCosts(DynaPlex::Policy policy) {
			//If this flag is set, it means that the memory of statemap is being used as part of an 
			//optimal policy. Using this same memory now for setting the optimal costs will lead to 
			//strange errors, hence better throw an error now. 
			if (exact_policy_exported)
				throw DynaPlex::Error("Illegal call to ExactPolicy::ComputeCosts: Exact policy has allready been exported. ");
			statemap.reserve(max_states);
			action_states.reserve(max_states);
			CreateStateMap();
			SetActions(policy);
			DetermineTransitions();
			do {
				for (size_t i = 0; i < 10; i++)
					IterateValues();
				if (!policy)
				{
					UpdateActionsForValues();
					DetermineTransitions();
				}
				IterateValues();
				CheckConvergence();
			} while (maxChange > 0.0001);
			if (!policy)
				exact_policy_computed = true;
			else
				exact_policy_computed = false;

			for (int64_t index = 0; index < action_states.size(); index++)
			{
				auto& state = action_states.at(index);
				auto& value = GetStateValue(state.state);
				if (value.state_index != index)
					throw DynaPlex::Error("Issue with state index");

			}

			return currentCost;
		}


	};

	class ExactPolicy : public PolicyInterface {
	public:
		// Constructor that accepts a shared pointer to the ExactSolver::Impl
		ExactPolicy(std::shared_ptr<ExactSolver::Impl> solverImpl)
			: impl(solverImpl) {
			if (!impl) {
				throw DynaPlex::Error("ExactPolicy constructor requires a valid solver implementation");
			}
		}

		// Override TypeIdentifier() to provide the type of the policy
		virtual std::string TypeIdentifier() const override {
			return "ExactPolicy";
		}


		virtual std::vector<int64_t> GetPromisingActions(const DynaPlex::dp_State& dp_state, int64_t num_actions) const override {
			return { 0 };
		}

		// Override GetConfig() to return configuration details
		virtual const DynaPlex::VarGroup& GetConfig() const override {
			static DynaPlex::VarGroup config;
			config.Add("NumStates", static_cast<int64_t>(impl->action_states.size()));
			config.Add("Objective", 10);

			return config;
		}

		// Override SetAction to set actions for all provided trajectories
		virtual void SetAction(std::span<Trajectory> trajectories) const override {
			for (auto& traj : trajectories) {
				const DynaPlex::StateCategory& cat = traj.Category;


				if (cat.IsAwaitAction())
				{
					auto& value = impl->GetStateValueTS(traj.GetState());
					traj.NextAction = impl->action_states[value.state_index].current_action;
					auto& retrieved = impl->action_states[value.state_index].state;

					if (!impl->mdp->IsAllowedAction(traj.GetState(), traj.NextAction)) {
						throw DynaPlex::Error("Illegal action retrieved from solver.");
					}
				}
				else
					throw DynaPlex::Error("OptimalPolicy: All actions must be awaiting action to set actions.");
			}
		}
		// Destructor
		virtual ~ExactPolicy() {}

	private:
		std::shared_ptr<ExactSolver::Impl> impl;
	};


	ExactSolver::ExactSolver(const System& system, MDP mdp, const DynaPlex::VarGroup& config)
		: pImpl(std::make_unique<Impl>(system, mdp, config)) {
	}

	double ExactSolver::ComputeCosts(DynaPlex::Policy policy) {
		return pImpl->ComputeCosts(policy);
	}

	DynaPlex::Policy ExactSolver::GetOptimalPolicy() {
		//compute the optimal policy if not allready computed. 
		if (!pImpl->exact_policy_computed)
			ComputeCosts();
		//We set this flag, so that trying to compute the costs for another policy after 
		//exporting the optimal policy raises an error. (Note that the memory of the ActionStates is shared between policy
		// and optimizer..)
		pImpl->exact_policy_exported = true;
		return std::make_shared<ExactPolicy>(pImpl);
	}
}