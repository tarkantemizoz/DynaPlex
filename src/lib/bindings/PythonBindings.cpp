#include <pybind11/pybind11.h>
#include "dynaplex/pythonparams.h"
#include "dynaplex/errors.h"
#include "dynaplex/neuralnetworktrainer.h"

namespace py = pybind11;

void process(py::object& obj)
{
	auto pars = DynaPlex::PythonParams(obj);
	pars.Print();
}
void process(py::kwargs& kwargs)
{
	auto& obj = static_cast<py::object&>(kwargs);
	process(obj);
}

void testPytorch()
{
	DynaPlex::NeuralNetworkTrainer trainer{};
	trainer.writeidentifier();
}

py::dict get()
{
	DynaPlex::Params distprops{
			{"type","geom"},
		{"mean",5} };

	return DynaPlex::PythonParams(std::move(distprops));
}


PYBIND11_MODULE(DP_Bindings, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring
	m.def("get", &get, "gets some parameters");
	m.def("process", py::overload_cast<py::kwargs&>(&process), "Processes kwargs");
	m.def("process", py::overload_cast<py::object&>(&process), "Processes dict");
	m.def("testPyTorch", &testPytorch, "tests pytorch availability");

}