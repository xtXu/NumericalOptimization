#include <iostream>
#include <Eigen/Eigen>
#include <func.hpp>

using namespace numerical_optimization;

int main (int argc, char *argv[]) {
	std::cout << "hello world" << std::endl;	

	Eigen::VectorXd a(6);
	a << 1.0, 1.0, 3.0, 8.0, 1.0, 11.0;

	double b = Rosenbrock::at(a);
	Eigen::VectorXd g = Rosenbrock::grad(a);
	std::cout << b << std::endl;
	std::cout << g << std::endl;
	
	return 0;
}

