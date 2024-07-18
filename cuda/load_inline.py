import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world() {
    return "Hello world";
}
"""

my_module = load_inline(
    name = 'module',
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True
)

print(my_module.hello_world())

######
## OUTPUT from running on Docker container
######
# Using /root/.cache/torch_extensions/py310_cu121 as PyTorch extensions root...
# Creating extension directory /root/.cache/torch_extensions/py310_cu121/module...
# Emitting ninja build file /root/.cache/torch_extensions/py310_cu121/module/build.ninja...
# Building extension module module...
# Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
# [1/2] c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=module -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /opt/conda/lib/python3.10/site-packages/torch/include -isystem /opt/conda/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /opt/conda/lib/python3.10/site-packages/torch/include/TH -isystem /opt/conda/lib/python3.10/site-packages/torch/include/THC -isystem /opt/conda/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py310_cu121/module/main.cpp -o main.o
# [2/2] c++ main.o -shared -L/opt/conda/lib/python3.10/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o module.so
# Loading extension module module...
# Hello world