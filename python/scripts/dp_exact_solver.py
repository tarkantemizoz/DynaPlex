from dp import dynaplex

mdp = dynaplex.get_mdp(id="lost_sales",
                       p=4.0,
                       leadtime=2,
                       demand_dist={"type": "poisson", "mean": 5.0},
                       bound_order_size_to_max_system_inv=True)
#note that bound_order_size_to_max_system_inv=true is needed to reproduce zipkin, because otherwise the
#base-stock policy is effectively bounded by the order cap for individual orders.

base_policy = mdp.get_policy("base_stock")

exact_solver = dynaplex.get_exact_solver(mdp)
cost = exact_solver.compute_costs(base_policy)
print(cost)
exact_cost = exact_solver.compute_costs()
print(exact_cost)
exact_policy = exact_solver.get_optimal_policy()

comparer = dynaplex.get_comparer(mdp)
comparison = comparer.compare(base_policy, exact_policy)
result = [(item['mean']) for item in comparison]
print(result)