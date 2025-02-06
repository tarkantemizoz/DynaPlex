#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"



namespace DynaPlex::Models {
	namespace driver_assignment /*keep this in line with id below and with namespace name in header*/
	{

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("current_driver_list_conc", current_driver_list_conc);
			vars.Add("current_order_typelist", current_order_typelist);
			vars.Add("current_time", current_time);
			//add any other state variables. 
			return vars;
		}

		MDP::State MDP::GetState(const VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			// (is_busy, loc_X, loc_y, next_time_available, remaining work hours) concatenated
			vars.Get("current_driver_list_conc", state.current_driver_list_conc);
			// (pickup_loc_x, pickup_loc_y, deliver_loc_x, deliver_loc_y)
			vars.Get("current_order_typelist", state.current_order_typelist);
			vars.Get("current_time", state.current_time);
			//initiate any other state variables. 
			return state;
		}

		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;
			//Needs to update later:
			vars.Add("valid_actions", number_drivers);

			vars.Add("horizon_type", "infinite"); // default
			// if finite: decision periods must be discrete time periods
			// since we have exponential interarrival times, we have to choose infinite and set horizon length to stop

			return vars;
		}

		MDP::MDP(const VarGroup& config)
		{
			config.Get("number_drivers", number_drivers);
			config.Get("mean_order_per_time_unit", mean_order_per_time_unit);
			config.Get("max_x", max_x);
			config.Get("max_y", max_y);
			config.Get("max_working_hours", max_working_hours);
			config.Get("horizon_length", horizon_length);
			config.Get("cost_per_km", cost_per_km);
			config.Get("penalty_for_workhour_violation", penalty_for_workhour_violation);

		}

		MDP::State MDP::GetInitialState(DynaPlex::RNG& rng) const
		{

			State state{};
			state.cat = StateCategory::AwaitEvent(); // after this we will await an event
			state.current_time = 0.0;
			state.current_driver_list_conc.reserve(number_drivers * 5); // each driver has 5 attributes
			for (int64_t i = 0; i < number_drivers; i++) {
				state.current_driver_list_conc.push_back(0.0); // all drivers start idle ie. isBusy = 0.0
				state.current_driver_list_conc.push_back(rng.genUniform() * max_x + min_x); // initial random
				state.current_driver_list_conc.push_back(rng.genUniform() * max_y + min_y); // initial random
				state.current_driver_list_conc.push_back(state.current_time); // next time being available is initially now
				state.current_driver_list_conc.push_back(max_working_hours); // from config
			}

			state.current_order_typelist.resize(4, 0.0); // order with 4 attributes

			return state;
		}


		MDP::Event MDP::GetEvent(RNG& rng) const {
			// interarrival times b/w events/orders are exponentially distributed
			// we define this by modifying uniform dist as exponential distribution is not defined in dynaplex yet 
			double random_variable_for_interarrivaltime = rng.genUniform();
			double interarrivaltime = std::log(1 - random_variable_for_interarrivaltime) / (-mean_order_per_time_unit);
			double pickup_x = rng.genUniform() * max_x + min_x;
			double pickup_y = rng.genUniform() * max_y + min_y;
			double deliver_x = rng.genUniform() * max_x + min_x;
			double deliver_y = rng.genUniform() * max_y + min_y;
			//burada randomly create ettigim itemlardan bir event vectoru yaratiyorum. 
			std::vector<double> event_vec{ interarrivaltime, pickup_x, pickup_y, deliver_x, deliver_y };

			return event_vec;
		}

		double MDP::ModifyStateWithEvent(State& state, const Event& event) const
		{
			state.cat = StateCategory::AwaitAction(); // after this we will await an action

			double interarrivaltime = event[0];
			state.current_order_typelist[0] = event[1];
			state.current_order_typelist[1] = event[2];
			state.current_order_typelist[2] = event[3];
			state.current_order_typelist[3] = event[4];
			state.current_time += interarrivaltime;

			for (int64_t i = 0; i < number_drivers; i++) {// attributes of ith driver start from i*5 of the concatenated driver attributes


				if (state.current_driver_list_conc[i * 5] == 0.0) { // if it'd idle we just need to update next_time_avail to current_time
					state.current_driver_list_conc[i * 5 + 3] = state.current_time;
				}
				else if (state.current_driver_list_conc[i * 5] == 1.0 && state.current_driver_list_conc[i * 5 + 3] <= state.current_time) {
					// if it's busy and next_time_avail <= current_time
					// we also cover the equality, it finished at the same time with order arrival
					state.current_driver_list_conc[i * 5] = 0.0; // update to not busy since it has already finished before a new event arrival
					state.current_driver_list_conc[i * 5 + 3] = state.current_time; // update next_time_avail to current_time
				}
				// else we do nothing 
				// BECAUSE
				// if (busy and next_time_avail > current_time)
				// it remains busy and next_time_avail remains the same (some time in the future)


			}

			return 0.0;
		}

		double MDP::ModifyStateWithAction(MDP::State& state, int64_t action) const
		{

			state.cat = StateCategory::AwaitEvent(); // after this we will await an event


			double cost_normal = 0.0;
			double cost_overtime = 0.0;
			// abs_dist(driver loc, picup loc):
			double pick_up_distance_x = std::abs(state.current_driver_list_conc[action * 5 + 1] - state.current_order_typelist[0]);
			double pick_up_distance_y = std::abs(state.current_driver_list_conc[action * 5 + 2] - state.current_order_typelist[1]);
			// abs_dist(picup loc, deliver loc): 
			double delivery_distance_x = std::abs(state.current_order_typelist[0] - state.current_order_typelist[2]);
			double delivery_distance_y = std::abs(state.current_order_typelist[1] - state.current_order_typelist[3]);
			// total dist is driver to pickup to delivery
			double total_dist = pick_up_distance_x + pick_up_distance_y + delivery_distance_x + delivery_distance_y;
			double total_duration = total_dist; // since avg_speed is 60km/hr and we keep track of minutes

			if (state.current_driver_list_conc[action * 5] == 0.0) { // if idle

				state.current_driver_list_conc[action * 5] = 1.0; // driver becomes busy

			}


			// driver's new location after delivery:
			state.current_driver_list_conc[action * 5 + 1] = state.current_order_typelist[2];// driver_x = deliver_x
			state.current_driver_list_conc[action * 5 + 2] = state.current_order_typelist[3];// driver_x = deliver_y

			state.current_driver_list_conc[action * 5 + 3] += total_dist; // next_time_avail increases

			state.current_driver_list_conc[action * 5 + 4] -= total_duration; // rem work hours decreases

			if (state.current_driver_list_conc[action * 5 + 4] < 0.0) {
				// if rem_work_hours becomes negative we need to add penalty cost for distance in overtime 
				cost_overtime += penalty_for_workhour_violation * (-state.current_driver_list_conc[action * 5 + 4]);
			}

			// calculate cost of distance
			cost_normal += cost_per_km * (total_dist + state.current_driver_list_conc[action * 5 + 4]); // normal_dist = total_dist - over_time_dist


			// When to stop 
			if (state.current_time >= horizon_length) {
				state.current_time = 0.0;
				for (int64_t i = 0; i < number_drivers; i++) {
					state.current_driver_list_conc[i * 5] = 0.0; // all becomes idle
					// drivers stay put
					state.current_driver_list_conc[i * 5 + 3] = 0.0; // reset next_time_avail
					state.current_driver_list_conc[i * 5 + 4] = max_working_hours; // reset rem work hours
				}
			}

			return cost_normal + cost_overtime;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {

			return true;
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features)const {

			for (int64_t i = 0; i < number_drivers; i++)
			{
				features.Add(state.current_driver_list_conc[i * 5]); // is_busy
				features.Add(state.current_driver_list_conc[i * 5 + 1] / max_x); // loc_x NORMALIZED 
				features.Add(state.current_driver_list_conc[i * 5 + 2] / max_y); // loc_y NORMALIZED
				features.Add(state.current_driver_list_conc[i * 5 + 3]); // next_time_available  ?Normalize?
				features.Add(state.current_driver_list_conc[i * 5 + 4] / max_working_hours); // rem_working_hours NORMALIZED
			}


			features.Add(state.current_order_typelist[0] / max_x); // pickup_loc_x NORMALIZED
			features.Add(state.current_order_typelist[1] / max_y); // pickup_loc_y NORMALIZED
			features.Add(state.current_order_typelist[2] / max_x); // deliver_loc_x NORMALIZED
			features.Add(state.current_order_typelist[3] / max_y); // deliver_loc_y NORMALIZED

			features.Add(state.current_time); // ? Normalize ?
		}

		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{//Here, we register any custom heuristics we want to provide for this MDP.	
		 //On the generic DynaPlex::MDP constructed from this, these heuristics can be obtained
		 //in generic form using mdp->GetPolicy(VarGroup vars), with the id in var set
		 //to the corresponding id given below.
			registry.Register<GreedyPolicy>("greedy_policy",
				//greedy policy, en yakin driver assign edilir ordera
				"This policy will take the closest driver to the order when asked for an action. ");
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			//this typically works, but state.cat must be kept up-to-date when modifying states. 
			return state.cat;
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel("driver_assignment", "pickup delivery problem with no order rejection", registry);
		}
	}
}


