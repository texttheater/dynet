#define BOOST_TEST_MODULE TEST_NODES

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>
#include <dynet/jacobian.h>
#include <boost/test/unit_test.hpp>
#include <stdexcept>

using namespace dynet;
using namespace dynet::expr;
using namespace std;


struct BuilderTest {
  BuilderTest() {
    // initialize if necessary
    if (default_device == nullptr) {
      for (auto x : {"BuilderTest", "--dynet-mem", "10"}) {
        av.push_back(strdup(x));
      }
      char **argv = &av[0];
      int argc = av.size();
      dynet::initialize(argc, argv);
    }
    std::vector<float> param1_vals = {1.1f, -2.2f, 3.3f};
    std::vector<float> param_square1_vals = {1.1f, 2.2f, 3.4f, 1.2f, 2.5f, 3.2f, 5.3f, 2.3f, 3.3f};
    param1 = mod.add_parameters({3});
    TensorTools::SetElements(param1.get()->values, param1_vals);
    param_square1 = mod.add_parameters({3, 3});
    TensorTools::SetElements(param_square1.get()->values, param_square1_vals);
  }
  ~BuilderTest() {
    for (auto x : av) free(x);
  }
  std::vector<char*> av;
  dynet::Model mod;
  dynet::Parameter param1, param_square1;
};

// define the test suite
BOOST_FIXTURE_TEST_SUITE(builder_test, BuilderTest);


// Expression operator-(const Expression& x);
BOOST_AUTO_TEST_CASE( jacobian_multiply ) {
  JacobianBuilder jac;
  dynet::ComputationGraph cg;
  Expression x1 = parameter(cg, param1);
  Expression x2 = parameter(cg, param_square1);
  Expression y = x2 * x1;
  Expression j = jac.calc_jacobian(x1, y);
  Expression z = sum_elems(j);
  BOOST_CHECK(check_grad(mod, z, 0));
}

BOOST_AUTO_TEST_SUITE_END()
