
#include "dynet/dynet.h"
#include "dynet/jacobian.h"

#include <set>

using namespace dynet;
using namespace std;

bool JacobianBuilder::find_path(ComputationGraph & cg, VariableIndex start, VariableIndex now,
                                set<VariableIndex> & visited, set<VariableIndex> & on_path) {
  if(now < start || visited.find(now) != visited.end())
    return false;
  visited.insert(now);
  bool my_on_path = (start == now);
  if(!my_on_path)
    for(auto & node : cg.nodes[now]->args)
      my_on_path = my_on_path || find_path(cg, start, node, visited, on_path);
  if(my_on_path)
    on_path.insert(now);
  return my_on_path;
}

VariableIndex JacobianBuilder::calc_jacobian(ComputationGraph & cg, VariableIndex start, VariableIndex end) {
  DYNET_ARG_CHECK(start < end, "The end of a Jacobian calculation must occur after the start.");
  set<VariableIndex> visited, needs;
  bool on_path = find_path(cg, start, end, visited, needs);
  DYNET_ARG_CHECK(on_path, "Could not find path from start to end in JacobianBuilder");
  VariableIndex invalid = (VariableIndex)-1;
  vector<VariableIndex> jac_nodes(end+1, invalid);
  DYNET_ARG_CHECK(cg.nodes[start]->dim.nd == 1, "Start of Jacobian computation is not a vector, with length " << cg.nodes[start]->dim);
  jac_nodes[start] = cg.add_function<IdentityMatrix>(cg.nodes[start]->dim[0]);
  VariableIndex next = start;
  while(next < end) {
    ++next;
    // Get the predecessors
    bool has_jac_pred = false;
    vector<VariableIndex> pred(cg.nodes[next]->args.size()), jac_pred(cg.nodes[next]->args.size());
    for(size_t j = 0; j < cg.nodes[next]->args.size(); ++j) {
      pred[j] = cg.nodes[next]->args[j];
      jac_pred[j] = jac_nodes[pred[j]];
      has_jac_pred = has_jac_pred || (jac_pred[j] != invalid);
    }
    if(!has_jac_pred) continue;
    // Do the actual computation
    NodeType my_type = cg.nodes[next]->node_type();
    if (my_type == NodeType::MatrixMultiply) {
      DYNET_ASSERT(pred.size() == 2, "Size of MatrixMultiply predecessors not 2");
      if(jac_pred[0] == invalid)
        jac_nodes[next] = cg.add_function<MatrixMultiply>({pred[0], jac_pred[1]});
      else if(jac_pred[1] == invalid)
        jac_nodes[next] = cg.add_function<MatrixMultiply>({jac_pred[0], pred[1]});
      else
        jac_nodes[next] = cg.add_function<Sum>({ 
          cg.add_function<MatrixMultiply>({pred[0], jac_pred[1]}),
          cg.add_function<MatrixMultiply>({jac_pred[0], pred[1]})
        });
    } else {
      DYNET_RUNTIME_ERR("Node "<<next<<" required for Jacobian calculation but not implemented yet");
    }
  }
  DYNET_ASSERT(jac_nodes[next] != nullptr, "Got to a null pointer");
  return jac_nodes[next];
}

Expression JacobianBuilder::calc_jacobian(const Expression & start, const Expression & end) {
  return Expression(start.pg, calc_jacobian(*start.pg, start.i, end.i));
}
