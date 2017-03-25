#ifndef DYNET_JACOBIAN_H
#define DYNET_JACOBIAN_H

#include "dynet/expr.h"

namespace dynet {

class JacobianBuilder {
public:
  JacobianBuilder() { }

  /**
  * /brief Calculate the Jacobian
  * /detailed Calculates the jacobian matrix of end given start
  */
  dynet::expr::Expression calc_jacobian(const dynet::expr::Expression & start, const dynet::expr::Expression & end);
  VariableIndex calc_jacobian(ComputationGraph & cg, VariableIndex start, VariableIndex end);

  bool find_path(ComputationGraph & cg, VariableIndex start, VariableIndex now,
                 std::set<VariableIndex> & visited, std::set<VariableIndex> & on_path);
  

};

}

#endif
