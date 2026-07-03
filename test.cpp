#include <iostream>

#include <pybind11/pybind11.h>

double foo(const double & x){
  return x+1;
}


PYBIND11_MODULE(simple_module, m){


  m.def("myfoo", &foo)
  
}
