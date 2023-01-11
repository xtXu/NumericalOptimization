/**
 * @file
 * @brief The Linear-search Steepest Gradient Descent (LSGD) optimization
 */

#include <Eigen/Eigen>
#include <iostream>

namespace numerical_optimization {

typedef double (*func_value)(
    const Eigen::VectorXd &x); // The func handler for calculating the value
typedef Eigen::VectorXd (*func_grad)(
    const Eigen::VectorXd &x); // The func handler for calculating the gradient

/**
 * @class LsgdParam
 * @brief The param used for LSGD method
 *
 */
struct LsgdParam {
  double tolerance; // the tolerance of the final gradient's norm
  double armijo_c;  // the parameter c for armijo line search
  bool info;        // whether print the info of each iteration
};

/**
 * @class LsgdOpt
 * @brief Linear-search Steepest Gradient Descent (LSGD) optimizer 
 *
 */
class LsgdOpt {
public:
  static void optimize(Eigen::VectorXd &x, double &f, func_value fx_evaluate,
                       func_grad gfx_evaluate, LsgdParam &lsgd_param);

private:
  static void line_search_armijo(Eigen::VectorXd &x, const double &fx,
                                 const Eigen::VectorXd &gx,
                                 const Eigen::VectorXd &d,
                                 func_value fx_evaluate, double c);
};

} // namespace numerical_optimization
