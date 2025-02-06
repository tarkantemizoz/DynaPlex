#include "dynaplex/models/registrationmanager.h"
#include "dynaplex/registry.h"

namespace DynaPlex::Models {
	//forward declarations of the registration functions of MDPs:
	namespace lost_sales {
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
	namespace multi_item_sla {
		void Register(DynaPlex::Registry&);
	}
	namespace driver_assignment {
		void Register(DynaPlex::Registry&);
	}
	void RegistrationManager::RegisterAll(DynaPlex::Registry& registry) {
		lost_sales::Register(registry);
		perishable_systems::Register(registry);
		Zero_Shot_Lost_Sales_Inventory_Control::Register(registry);
		random_leadtimes::Register(registry);
		multi_item_sla::Register(registry);
		driver_assignment::Register(registry);
	}
}
