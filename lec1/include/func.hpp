/**
 * @file
 * @brief The Rosenbrock function
 */

#include <Eigen/Eigen>
#include <iostream>

namespace numerical_optimization {

/**
 * @class Rosenbrock
 * @brief Calculate the value and gradient for N dimension Rosenbrock function.
 *
 */
class Rosenbrock {
public:
  static double at(const Eigen::VectorXd &x);
  static Eigen::VectorXd grad(const Eigen::VectorXd &x);
};

/**
 * @brief Calculate the value of Rosenbrock function.
 *
 * @param x The input x. The dimension is depends on the Eigen vector dimension.
 * @return The value of Rosenbrock function at x.
 */
inline double Rosenbrock::at(const Eigen::VectorXd &x) {
  double ret = 0;
  int n = x.size();

	// Check if the dimension is an even number
	if (n%2 != 0) {
		std::cerr << "The dimension of input x must be an even number !" << std::endl;
		exit(1);
	}

	// calculate the value
  for (int i = 0; i < n / 2; i++) {
    ret +=
        100 * pow((pow(x(2 * i), 2) - x(2 * i + 1)), 2) + pow(x(2 * i) - 1, 2);
  }

  return ret;
}

/**
 * @brief Calculate the gradient of Rosenbrock function.
 *
 * @param x The input x.
 * @return The gradient of Rosenbrock func at x. The dimension is same as x.
 */
inline Eigen::VectorXd Rosenbrock::grad(const Eigen::VectorXd &x) {
  int n = x.size();

	// Check if the dimension is an even number
	if (n%2 != 0) {
		std::cerr << "The dimension of input x must be an even number !" << std::endl;
		exit(1);
	}

  Eigen::VectorXd grad(n);

	// calculate the gradient 
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
