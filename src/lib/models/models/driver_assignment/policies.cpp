#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
namespace DynaPlex::Models {
	namespace driver_assignment /*keep this namespace name in line with the name space in which the mdp corresponding to this policy is defined*/
	{

		//MDP and State refer to the specific ones defined in current namespace
		GreedyPolicy::GreedyPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			//Here, you may initiate any policy parameters.
		}

		int64_t GreedyPolicy::GetAction(const MDP::State& state) const
		{

			std::vector<double> distances;
			for (int64_t i = 0; i < mdp->number_drivers; i++) {
				double dist_x = std::abs(state.current_driver_list_conc[i * 5 + 1] - state.current_order_typelist[0]);
				double dist_y = std::abs(state.current_driver_list_conc[i * 5 + 2] - state.current_order_typelist[1]);
				distances.push_back(dist_x + dist_y);
			}
			auto minDistance = std::min_element(distances.begin(), distances.end()); // smallest discard
			int64_t best_driver_index = std::distance(distances.begin(), minDistance); // index with smallest expected discard

			return best_driver_index;
		}
	}
}