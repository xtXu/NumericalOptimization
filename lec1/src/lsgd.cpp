#include <lsgd.hpp>

namespace numerical_optimization {

/**
 * @brief The main process of opimization.
 *
 * @param x Input as the initial x0, output as the x*.
 * @param f Output as the final minimum function value.
 * @param fx_evaluate Function handler for calculating the value.
 * @param gfx_evaluate Function handler for calculating the gradient.
 * @param lsgd_param The param struct for the optimizer.
 */
void LsgdOpt::optimize(Eigen::VectorXd &x, double &f, func_value fx_evaluate,
                       func_grad gfx_evaluate, LsgdParam &lsgd_param) {
  Eigen::VectorXd g; // gradient
  Eigen::VectorXd d; // direction

  // calculate the initial value, gradient and direction
  f = fx_evaluate(x);
  g = gfx_evaluate(x);
  d = -g;

  // print the initial state
  if (lsgd_param.info) {
    std::cout << "initial x: " << x.transpose() << std::endl;
    std::cout << "initial f: " << f << std::endl;
    std::cout << "initial g: " << g.transpose() << std::endl;
    std::cout << "initial d: " << d.transpose() << std::endl;
  }

  int iter_cnt = 0;
  // iteration if the norm of gradient is over the tolerance
  while (g.norm() >= lsgd_param.tolerance) {
    // use Armijo line-search to obtain the next x_k+1
    line_search_armijo(x, f, g, d, fx_evaluate, lsgd_param.armijo_c);
    // calculate the state of x_k+1
    f = fx_evaluate(x);
    g = gfx_evaluate(x);
    d = -g;
    iter_cnt++;

    // print the iteration state
    if (lsgd_param.info) {
      std::cout << "iteration " << iter_cnt << " :" << std::endl;
      std::cout << "x: " << x.transpose() << std::endl;
      std::cout << "f: " << f << std::endl;
      std::cout << "g: " << g.transpose() << std::endl;
      std::cout << "d: " << d.transpose() << std::endl;
    }
  }

  // print the result
  if (lsgd_param.info) {
    std::cout << "x* : " << x.transpose() << std::endl;
    std::cout << "The minimum : " << f << std::endl;
  }
}

/**
 * @brief The Armijo line-search method.
 *
 * @param x  Input as the x_k in current iteration, output as the new x_k+1
 * after stepping.
 * @param fx f(x_k)
 * @param gx The gradient of f(x) at x_k.
 * @param d The direction of stepping.
 * @param fx_evaluate The func handler for calculating the f(x).
 * @param c The param c in Armijo method.
 */
void LsgdOpt::line_search_armijo(Eigen::VectorXd &x, const double &fx,
                                 const Eigen::VectorXd &gx,
                                 const Eigen::VectorXd &d,
                                 func_value fx_evaluate, double c) {
  double alpha = 1.0; // initial step size
  double fx_new;
  double k_threshold = -c * d.dot(gx); // d.dot(gx) is the derivatives of
                                       // phi(0), phi(alpha)=f(x_k+alpha*d)
  Eigen::VectorXd x0 = x;

  x = x0 + alpha * d;
  fx_new = fx_evaluate(x);

  while (fx - fx_new < k_threshold * alpha) {
		// if the alpha does not meet Armijo condition, decrease it
    alpha /= 2;
		// calculate new x and new fx
    x = x0 + alpha * d;
    fx_new = fx_evaluate(x);
  }
}

} // namespace numerical_optimization
