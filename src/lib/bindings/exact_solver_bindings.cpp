#include "dynaplex/exactsolver.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "dynaplex/vargroup.h"
#include "vargroupcaster.h"
void define_exact_solver_bindings(py::module_& m) {
    using namespace DynaPlex::Algorithms;

    py::class_<ExactSolver>(m, "exact_solver")
        .def("compute_costs", &ExactSolver::ComputeCosts,
            py::arg("policy") = nullptr,
            "Computes exact return (costs/rewards) for the given policy or computes exact optimal costs if no policy is provided.")
        .def("get_optimal_policy", &ExactSolver::GetOptimalPolicy,
            "Returns the optimal policy. Will reuse the computed policy from `compute_costs` if already available, otherwise initiates the exact solution process.");
}