//
// Created by Davi Colli Tozoni on 7/30/18.
//

#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/LinearElasticityWithContact.hh>

int main(int argc, char** argv) {
    SmoothPositiveMax function(0.1);

    std::cout << "Verifying values for: " << std::endl;
    std::cout << "  value(-10.0): " << function.value(-10.0) << std::endl;
    std::cout << "  value(-1.0): " << function.value(-1.0) << std::endl;
    std::cout << "  value(-0.1): " << function.value(-0.1) << std::endl;
    std::cout << "  value(-0.01): " << function.value(-0.01) << std::endl;
    std::cout << "  value(0.01): " << function.value(0.01) << std::endl;
    std::cout << "  value(0.1): " << function.value(0.1) << std::endl;
    std::cout << "  value(1.0): " << function.value(1.0) << std::endl;
    std::cout << "  value(10.0): " << function.value(10.0) << std::endl;
    std::cout << std::endl;

    std::cout << "Verifying derivatives for: " << std::endl;
    std::cout << "  derivative(-10.0): " << function.derivative(-10.0) << std::endl;
    std::cout << "  derivative(-1.0): " << function.derivative(-1.0) << std::endl;
    std::cout << "  derivative(-0.1): " << function.derivative(-0.1) << std::endl;
    std::cout << "  derivative(-0.01): " << function.derivative(-0.01) << std::endl;
    std::cout << "  derivative(0.01): " << function.derivative(0.01) << std::endl;
    std::cout << "  derivative(0.1): " << function.derivative(0.1) << std::endl;
    std::cout << "  derivative(1.0): " << function.derivative(1.0) << std::endl;
    std::cout << "  derivative(10.0): " << function.derivative(10.0) << std::endl;

    return 0;
}