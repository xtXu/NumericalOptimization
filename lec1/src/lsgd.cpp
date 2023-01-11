#include <lsgd.hpp>

namespace numerical_optimization {

void LsgdOpt::optimize(Eigen::VectorXd &x, double &f, func_value fx_evaluate,
                       func_grad gfx_evaluate, LsgdParam &lsgd_param) {
  Eigen::VectorXd g; // gradient
  Eigen::VectorXd d; // direction

  f = fx_evaluate(x);
  g = gfx_evaluate(x);
  d = -g;

  if (lsgd_param.info) {
    std::cout << "initial x: " << x.transpose() << std::endl;
    std::cout << "initial f: " << f << std::endl;
    std::cout << "initial g: " << g.transpose() << std::endl;
    std::cout << "initial d: " << d.transpose() << std::endl;
  }

  int iter_cnt = 0;
  while (g.norm() >= lsgd_param.tolerance) {
    line_search_armijo(x, f, g, d, fx_evaluate, lsgd_param.armijo_c);
    f = fx_evaluate(x);
    g = gfx_evaluate(x);
    d = -g;
    iter_cnt++;

    if (lsgd_param.info) {
      std::cout << "iteration " << iter_cnt << " :" << std::endl;
      std::cout << "x: " << x.transpose() << std::endl;
      std::cout << "f: " << f << std::endl;
      std::cout << "g: " << g.transpose() << std::endl;
      std::cout << "d: " << d.transpose() << std::endl;
    }
  }

  if (lsgd_param.info) {
    std::cout << "x* : " << x.transpose() << std::endl;
    std::cout << "The minimum : " << f << std::endl;
  }
}

void LsgdOpt::line_search_armijo(Eigen::VectorXd &x, const double &fx,
                                 const Eigen::VectorXd &gx,
                                 const Eigen::VectorXd &d,
                                 func_value fx_evaluate, double c) {
  double alpha = 1.0;
  double fx_new;
  double k_threshold = -c * d.dot(gx);
  Eigen::VectorXd x0 = x;

  x = x0 + alpha * d;
  fx_new = fx_evaluate(x);

  while (fx - fx_new < k_threshold * alpha) {
    alpha /= 2;
    x = x0 + alpha * d;
    fx_new = fx_evaluate(x);
  }
}

} // namespace numerical_optimization
