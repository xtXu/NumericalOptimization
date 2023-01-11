#include <Eigen/Eigen>
#include <iostream>

namespace numerical_optimization {

typedef double (*func_value)(const Eigen::VectorXd &x);
typedef Eigen::VectorXd (*func_grad)(const Eigen::VectorXd &x);

struct LsgdParam {
  double tolerance;
  double armijo_c;
	bool info;
};

// Linear-search Steepest Gradient Descent
class LsgdOpt {
public:
  static void optimize(Eigen::VectorXd &x, double &f, func_value fx_evaluate,
                       func_grad gfx_evaluate, LsgdParam &lsgd_param);

private:
  static void line_search_armijo(Eigen::VectorXd &x, const double &fx,
                          const Eigen::VectorXd &gx, const Eigen::VectorXd &d,
                          func_value fx_evaluate, double c);
};

} // namespace numerical_optimization
