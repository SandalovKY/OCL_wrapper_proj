#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>

#include "AxpyCPU.h"
#include "AxpyGPU.h"

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
	float a{ 1.4 };

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

	int64_t size{ 0 };
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
	}

	return EXIT_FAILURE;
}