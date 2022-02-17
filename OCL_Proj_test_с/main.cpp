#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>
#include <random>

#include "AxpyCPU.h"
#include "AxpyGPU.h"
#include "MatMult.h"

#include "DevWorker.h"
#include "GpuTask.h"

#define NAME_LENGTH 128
#define O_SIZE 1000

namespace
{
	std::string NVidia_platform{ "NVIDIA CUDA" };
	std::string NVidia_device{ "NVIDIA GeForce GTX 1650" };

	std::string AMD_platform{ "AMD Accelerated Parallel Processing" };
	std::string AMD_device{ "gfx902" };

	char const* kernelSrc = "__kernel void operation(__global long * inout,		\n"
							"						 const unsigned long count) \n"
							"{													\n"
							"	int index = get_global_id(0);					\n"
							"	int group = get_group_id(0);					\n"
							"	int lcl = get_local_id(0);						\n"
							"	printf(\"Hello from %i block, %i thread (global id: %i)\\n\", group, lcl, index); \n"
							"	if (index < count)								\n"
							"		inout[index] = inout[index] + index;		\n"
							"}													\n";
}

int testSaxpy(size_t size, size_t incx, size_t incy)
{
	std::cout << "\nVector Size: " << size << '\n';
	std::cout << "Saxpy times:\n";
	std::vector<float> y(size, 1.);
	std::vector<float> x(size, 1.);
	float a{ 1.4f };

	// Start CPU axpy
	double start = omp_get_wtime();
	my::_axpy(size, a, x, incx, y, incy);
	double end = omp_get_wtime() - start;
	std::cout << "\nTime for CPU: " << end << std::endl;

	for (const auto& vecEl : y)
	{
		if (std::abs(vecEl - 2.4) > 0.000001)
		{
			std::cout << "Incorrect output for the CPU faxpy\n";
			return EXIT_FAILURE;
		}
	}

	start = omp_get_wtime();
	my::saxpy_omp(size, 2, x, incx, y, incy);
	end = omp_get_wtime() - start;
	std::cout << "\nTime for CPU with OMP:" << end << std::endl;

	for (const auto& vecEl : y)
	{
		if (std::abs(vecEl - 4.4) > 0.000001)
		{
			std::cout << "Incorrect output for the OMP CPU faxpy\n";
			return EXIT_FAILURE;
		}
	}

	my::saxpy_gpu(size, static_cast<cl_float>(1), x, incx, y, incy, AMD_device.c_str());

	for (const auto& vecEl : y)
	{
		if (std::abs(vecEl - 5.4) > 0.000001)
		{
			std::cout << "Incorrect output for the AMD GPU faxpy\n";
			return EXIT_FAILURE;
		}
	}

	my::saxpy_gpu(size, static_cast<cl_float>(1), x, incx, y, incy, NVidia_device.c_str());

	for (const auto& vecEl : y)
	{
		if (std::abs(vecEl - 6.4) > 0.000001)
		{
			std::cout << "Incorrect output for the NVidia GPU faxpy\n";
			return EXIT_FAILURE;
		}
	}
	std::cout << "\n-------------------------------------\n";
	return EXIT_SUCCESS;
}

int testDaxpy(size_t size, size_t incx, size_t incy)
{
	std::cout << "\nVector Size: " << size << '\n';
	std::cout << "Daxpy times:\n";
	std::vector<double> y(size, 1.);
	std::vector<double> x(size, 1.);
	double a{ 1.4 };

	// Start CPU axpy
	double start = omp_get_wtime();
	my::_axpy(size, a, x, incx, y, incy);
	double end = omp_get_wtime() - start;
	std::cout << "\nTime for CPU: " << end << std::endl;

	for (const auto& vecEl : y)
	{
		if (std::abs(vecEl - 2.4) > 0.000001)
		{
			std::cout << "Incorrect output for the CPU faxpy\n";
			return EXIT_FAILURE;
		}
	}

	start = omp_get_wtime();
	my::daxpy_omp(size, 2, x, incx, y, incy);
	end = omp_get_wtime() - start;
	std::cout << "\nTime for CPU with OMP: " << end << std::endl;

	for (const auto& vecEl : y)
	{
		if (std::abs(vecEl - 4.4) > 0.000001)
		{
			std::cout << "Incorrect output for the OMP CPU faxpy\n";
			return EXIT_FAILURE;
		}
	}

	my::daxpy_gpu(size, static_cast<cl_float>(1), x, incx, y, incy, AMD_device.c_str());

	for (const auto& vecEl : y)
	{
		if (std::abs(vecEl - 5.4) > 0.000001)
		{
			std::cout << "Incorrect output for the AMD GPU faxpy\n";
			return EXIT_FAILURE;
		}
	}

	my::daxpy_gpu(size, static_cast<cl_float>(1), x, incx, y, incy, NVidia_device.c_str());

	for (const auto& vecEl : y)
	{
		if (std::abs(vecEl - 6.4) > 0.000001)
		{
			std::cout << "Incorrect output for the NVidia GPU faxpy\n";
			return EXIT_FAILURE;
		}
	}

	std::cout << "\n-------------------------------------\n";
	return EXIT_SUCCESS;
}

int main() {

	
	/*int64_t size{0};
	while (size <= 0)
	{
		std::cout << "Array syze: ";
		std::cin >> size;
	}
	int64_t incy{ 0 };
	int64_t incx{ 0 };

	while (incy <= 0 || incx <= 0)
	{
		std::cout << "IncY: ";
		std::cin >> incy;

		std::cout << "IncX: ";
		std::cin >> incx;
	}

	std::cout << "\nTesting saxpy:\n";
	for (int i = 0; i < 2; ++i)
	{
		testSaxpy(200000000, incx, incy);
	}
	std::cout << "\nTesting daxpy:\n";
	for (int i = 0; i < 2; ++i)
	{
		testDaxpy(100000000, incx, incy);
	}*/
	

	size_t X, Y, Z;
	X = 1024; Y = 1024; Z = 1024;

	std::vector<cl_int> matrA(Z * Y, 1);
	std::vector<cl_int> matrB(Y * X, 2);

	for (auto& matrEl : matrA)
	{
		matrEl = std::rand();
	}

	for (auto& matrEl : matrB)
	{
		matrEl = std::rand();
	}

	/*double start = omp_get_wtime();
	const auto resMatr = matMultCpu(matrA, matrB, Z, Y, X);
	start = omp_get_wtime() - start;
	std::cout << "Res MatMult time: " << start << std::endl;*/


	std::cout << "\nWithout using shared memory:\n";
	const auto resGpuMatr = matMultGpu(matrA, matrB, Z, Y, X, NVidia_device.c_str());
	const auto resGpuMatrAMD = matMultGpu(matrA, matrB, Z, Y, X, AMD_device.c_str());

	std::cout << "\nUsing shared memory:\n";
	const auto resGpuMatr1 = matMultGpu(matrA, matrB, Z, Y, X, NVidia_device.c_str(), true);
	const auto resGpuMatrAMD1 = matMultGpu(matrA, matrB, Z, Y, X, AMD_device.c_str(), true);

	/*for (int i = 0; i < resMatr.size(); ++i)
	{
		if (resMatr[i] != resGpuMatr[i])
		{
			std::cout << "Error\n";
			return EXIT_FAILURE;
		}
	}*/

	/*start = omp_get_wtime();
	const auto resMatr4 = matMultCpuBlock(matrA, matrB, Z, Y, X);
	start = omp_get_wtime() - start;
	std::cout << "Res MatMultBlock time: " << start << std::endl;

	for (int i = 0; i < resMatr.size(); ++i)
	{
		if (resMatr[i] != resMatr4[i])
		{
			std::cout << "Error\n";
			return EXIT_FAILURE;
		}
	}

	start = omp_get_wtime();
	const auto bTransp = transpMatr(matrB, X, Y);
	const auto resMatr2 = matMultCpuTransp(matrA, bTransp, Z, Y, X);
	start = omp_get_wtime() - start;
	std::cout << "Res MatMultTransp time: " << start << std::endl;

	for (int i = 0; i < resMatr.size(); ++i)
	{
		if (resMatr[i] != resMatr2[i])
		{
			std::cout << "Error\n";
			return EXIT_FAILURE;
		}
	}*/

	/*for (int i = 0; i < 5; ++i)
	{
		start = omp_get_wtime();
		const auto resMatr1 = matMultCpuOMP(matrA, matrB, X, Y, Z);
		start = omp_get_wtime() - start;
		std::cout << "Res MatMultOMP time: " << start << std::endl;

		for (int i = 0; i < resMatr.size(); ++i)
		{
			if (resMatr[i] != resMatr1[i])
			{
				std::cout << "Error\n";
				return EXIT_FAILURE;
			}
		}
	}*/

	for (int i = 0; i < 5; ++i)
	{
		auto start = omp_get_wtime();
		const auto bTransp1 = transpMatrOMP(matrB, X, Y);
		const auto resMatr1 = matMultCpuTranspOMP(matrA, bTransp1, X, Y, Z);
		start = omp_get_wtime() - start;
		std::cout << "\nRes MatMultOMPTransp time: " << start << std::endl;

		for (int i = 0; i < resMatr1.size(); ++i)
		{
			if (resGpuMatr[i] != resMatr1[i])
			{
				std::cout << "Error\n";
				return EXIT_FAILURE;
			}
		}
	}

	return EXIT_SUCCESS;
}