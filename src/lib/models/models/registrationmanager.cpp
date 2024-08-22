#include "dynaplex/models/registrationmanager.h"
#include "dynaplex/registry.h"

namespace DynaPlex::Models {
	//forward declarations of the registration functions of MDPs:
	namespace lost_sales {
		void Register(DynaPlex::Registry&);
	}
	namespace bin_packing {
		void Register(DynaPlex::Registry&);
	}
	namespace order_picking {
		void Register(DynaPlex::Registry&);
	}
	namespace perishable_systems {
		void Register(DynaPlex::Registry&);
	}
	namespace Zero_Shot_Lost_Sales_Inventory_Control {
		void Register(DynaPlex::Registry&);
	}
	namespace random_leadtimes {
		void Register(DynaPlex::Registry&);
	}
	void RegistrationManager::RegisterAll(DynaPlex::Registry& registry) {
		lost_sales::Register(registry);
		bin_packing::Register(registry);
		order_picking::Register(registry);
		perishable_systems::Register(registry);
		Zero_Shot_Lost_Sales_Inventory_Control::Register(registry);
		random_leadtimes::Register(registry);
	}
}
