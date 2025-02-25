#include <torch/extension.h>
#include "api.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("z_order_encode", &z_order_encode);
	m.def("z_order_decode", &z_order_decode);
	m.def("hilbert_encode", &hilbert_encode);
	m.def("hilbert_decode", &hilbert_decode);
}