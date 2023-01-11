#include <func.hpp>
#include <iostream>
#include <lsgd.hpp>

using namespace numerical_optimization;

int main(int argc, char *argv[]) {
  LsgdParam lsgd_param = {1e-6, 0.5, false};

	int n;

	std::cout << "Please enter the dimension of Rosenbrock function: "; 
	std::cin >> n;

  Eigen::VectorXd x = 3.0 * Eigen::VectorXd::Ones(n);

  double f;
  LsgdOpt::optimize(x, f, Rosenbrock::at, Rosenbrock::grad, lsgd_param);

	std::cout << "x*: " << x.transpose() << std::endl;
	std::cout << "minimum f: " << f << std::endl;

  return 0;
}
