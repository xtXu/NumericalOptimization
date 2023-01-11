#include <Eigen/Eigen>
#include <iostream>

namespace numerical_optimization {

class Rosenbrock {
public:
  static double at(const Eigen::VectorXd &x);
  static Eigen::VectorXd grad(const Eigen::VectorXd &x);
};

inline double Rosenbrock::at(const Eigen::VectorXd &x) {
  double ret = 0;
  int n = x.size();

	if (n%2 != 0) {
		std::cerr << "The dimension of input x must be an even number !" << std::endl;
		exit(1);
	}

  for (int i = 0; i < n / 2; i++) {
    ret +=
        100 * pow((pow(x(2 * i), 2) - x(2 * i + 1)), 2) + pow(x(2 * i) - 1, 2);
  }

  return ret;
}

inline Eigen::VectorXd Rosenbrock::grad(const Eigen::VectorXd &x) {
  int n = x.size();

	if (n%2 != 0) {
		std::cerr << "The dimension of input x must be an even number !" << std::endl;
		exit(1);
	}

  Eigen::VectorXd grad(n);
  for (int i = 0; i < n; i++) {
    double grad_i;
    if (i % 2 == 0) {
      grad_i = 200 * (pow(x(i), 2) - x(i + 1)) * 2 * x(i) + 2 * (x(i) - 1);
    } else {
      grad_i = -200 * (pow(x(i - 1), 2) - x(i));
    }
    grad[i] = grad_i;
  }

  return grad;
}

} // namespace numerical_optimization
