#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <algorithm>

namespace DynaPlex::Models {
	namespace windfarm /*keep this in line with id below and with namespace name in header*/
	{
		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;
			vars.Add("valid_actions", totalActions);
			vars.Add("horizon_type", "infinite");
			vars.Add("discount_factor", discount_factor);

			VarGroup diagnostics{};
			diagnostics.Add("num_windmills", num_windmills);
			diagnostics.Add("num_intervals", static_cast<int64_t>(interval_probs.size()));
			vars.Add("diagnostics", diagnostics);

			return vars;
		}

		MDP::MDP(const VarGroup& config)
		{
			config.Get("num_farms", num_farms);
			config.Get("windmills_per_farm", windmills_per_farm);
			num_windmills = num_farms * windmills_per_farm;

			// Degradation probabilities for Blue, Yellow, Orange.
			config.Get("degrade_probs", degrade_probs);   // {p1, p2, p3}
			config.Get("jump_red_probs", jump_red_probs);  // {q1, q2, q3}
			if (degrade_probs.size() != 3 || jump_red_probs.size() != 3)
				throw DynaPlex::Error("windfarm: degrade_probs and jump_red_probs must each have 3 entries (for Blue, Yellow, Orange).");

			config.Get("travel_cost", travel_cost);
			config.Get("maintenance_cost", maintenance_cost);
			config.Get("repair_cost", repair_cost);
			config.Get("red_penalty", red_penalty);

			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;

			totalActions = 1 + num_farms + windmills_per_farm;

			// --- Build the exact event model -------------------------------------------------
			// Per-state thresholds along the unit interval (measuring "amount of degradation"):
			//   Blue   : [0, q1) -> Red, [q1, q1+p1) -> Yellow, [q1+p1, 1) -> Blue
			//   Yellow : [0, q2) -> Red, [q2, q2+p2) -> Orange, [q2+p2, 1) -> Yellow
			//   Orange : [0, p3+q3) -> Red, [p3+q3, 1) -> Orange   (one step and jump both reach Red)
			//   Red    : absorbing
			const double q1 = jump_red_probs[Blue], p1 = degrade_probs[Blue];
			const double q2 = jump_red_probs[Yellow], p2 = degrade_probs[Yellow];
			const double q3 = jump_red_probs[Orange], p3 = degrade_probs[Orange];
			if (q1 + p1 > 1.0 + 1e-9 || q2 + p2 > 1.0 + 1e-9 || p3 + q3 > 1.0 + 1e-9)
				throw DynaPlex::Error("windfarm: transition probabilities for a level must not exceed 1.");

			std::vector<double> breakpoints = { 0.0, 1.0, q1, q1 + p1, q2, q2 + p2, p3 + q3 };
			std::sort(breakpoints.begin(), breakpoints.end());
			breakpoints.erase(std::unique(breakpoints.begin(), breakpoints.end(),
				[](double a, double b) { return std::abs(a - b) < 1e-12; }), breakpoints.end());

			for (auto& row : degradation_map)
				row.clear();

			for (size_t k = 0; k + 1 < breakpoints.size(); ++k)
			{
				const double lo = breakpoints[k];
				const double hi = breakpoints[k + 1];
				const double width = hi - lo;
				if (width <= 1e-12)
					continue; // drop zero-width intervals (coinciding thresholds)
				const double mid = 0.5 * (lo + hi); // a point representative of the whole sub-interval
				interval_probs.push_back(width);

				// Map (current health, this sub-interval) -> next health.
				degradation_map[Blue].push_back(mid < q1 ? Red : (mid < q1 + p1 ? Yellow : Blue));
				degradation_map[Yellow].push_back(mid < q2 ? Red : (mid < q2 + p2 ? Orange : Yellow));
				degradation_map[Orange].push_back(mid < p3 + q3 ? Red : Orange);
				degradation_map[Red].push_back(Red);
			}
		}

		// --- Actions -----------------------------------------------------------------------
		bool MDP::IsAllowedAction(const State& state, int64_t action) const
		{
			if (action == 0)
				return true; // idle is always allowed
			if (action <= num_farms)
			{
				// travel to farm (action-1); pointless to "travel" to the current farm.
				return (action - 1) != state.engineer_location;
			}
			// service local windmill at the current farm; only if it actually needs work.
			const int64_t local = action - num_farms - 1;
			const int64_t windmill = state.engineer_location * windmills_per_farm + local;
			return state.health[windmill] != Blue;
		}

		double MDP::ModifyStateWithAction(MDP::State& state, int64_t action) const
		{
			state.cat = StateCategory::AwaitEvent();

			if (action == 0)
				return 0.0; // idle

			if (action <= num_farms)
			{
				state.engineer_location = action - 1;
				return travel_cost;
			}

			const int64_t local = action - num_farms - 1;
			const int64_t windmill = state.engineer_location * windmills_per_farm + local;
			const int64_t h = state.health[windmill];
			state.health[windmill] = Blue;
			return (h == Red) ? repair_cost : maintenance_cost;
		}

		// --- Events (degradation) ----------------------------------------------------------
		MDP::Event MDP::GetEvent(RNG& rng) const
		{
			Event event;
			event.reserve(num_windmills);
			for (int64_t i = 0; i < num_windmills; ++i)
			{
				const double u = rng.genUniform();
				// locate the sub-interval containing u via its cumulative mass
				double cumulative = 0.0;
				int64_t idx = static_cast<int64_t>(interval_probs.size()) - 1;
				for (size_t k = 0; k < interval_probs.size(); ++k)
				{
					cumulative += interval_probs[k];
					if (u < cumulative)
					{
						idx = static_cast<int64_t>(k);
						break;
					}
				}
				event.push_back(idx);
			}
			return event;
		}

		std::vector<std::tuple<MDP::Event, double>> MDP::EventProbabilities() const
		{
			// Enumerate the cartesian product of per-windmill sub-intervals. Each windmill
			// is independent and shares the same interval distribution, so the joint
			// probability is the product of the per-windmill interval probabilities.
			const int64_t K = static_cast<int64_t>(interval_probs.size());
			std::vector<std::tuple<Event, double>> result;

			Event current(num_windmills, 0); // odometer over base-K digits
			bool done = false;
			while (!done)
			{
				double prob = 1.0;
				for (int64_t i = 0; i < num_windmills; ++i)
					prob *= interval_probs[current[i]];
				result.emplace_back(current, prob);

				// increment the odometer
				int64_t pos = 0;
				while (pos < num_windmills)
				{
					if (++current[pos] < K)
						break;
					current[pos] = 0;
					++pos;
				}
				if (pos == num_windmills)
					done = true;
			}
			return result;
		}

		double MDP::ModifyStateWithEvent(State& state, const MDP::Event& event) const
		{
			state.cat = StateCategory::AwaitAction();

			double cost = 0.0;
			for (int64_t i = 0; i < num_windmills; ++i)
			{
				state.health[i] = degradation_map[state.health[i]][event[i]];
				if (state.health[i] == Red)
					cost += red_penalty; // penalty for every windmill red during the upcoming period
			}
			return cost;
		}

		// --- States ------------------------------------------------------------------------
		MDP::State MDP::GetInitialState() const
		{
			State state{};
			state.cat = StateCategory::AwaitAction(); // engineer acts first
			state.health.assign(num_windmills, Blue);
			state.engineer_location = 0;
			return state;
		}

		MDP::State MDP::GetState(const VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("health", state.health);
			vars.Get("engineer_location", state.engineer_location);
			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("health", health);
			vars.Add("engineer_location", engineer_location);
			return vars;
		}

		// --- Features for the neural network -----------------------------------------------
		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const
		{
			// One-hot encode each windmill's health (4 categories) ...
			for (int64_t i = 0; i < num_windmills; ++i)
				for (int64_t h = 0; h < 4; ++h)
					features.Add(state.health[i] == h ? 1.0 : 0.0);
			// ... and the engineer's location (num_farms categories).
			for (int64_t f = 0; f < num_farms; ++f)
				features.Add(state.engineer_location == f ? 1.0 : 0.0);
		}

		// --- Boilerplate -------------------------------------------------------------------
		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{
			registry.Register<GreedyEngineerPolicy>("greedy_engineer",
				"Repairs/maintains the most urgent windmill at the current farm; otherwise travels "
				"to the farm with the most urgent windmill, or idles if everything is healthy.");
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				"windfarm",
				"Condition-based maintenance of windmills across multiple wind farms by a single travelling service engineer.",
				registry);
		}
	}
}
