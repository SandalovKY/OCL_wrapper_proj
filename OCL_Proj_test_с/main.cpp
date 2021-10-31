#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "axpy_cpu.h"

#include "DevWorker.h"

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
							//  "	int group = get_group_id(0);					\n"
							//  "	int lcl = get_local_id(0);						\n"
							//  "	printf(\"Hello from %i block, %i thread (global id: %i)\\n\", group, lcl, index); \n"
							"	if (index < count)								\n"
							"		inout[index] = inout[index] + index;		\n"
							"}													\n";

	char const* saxpy = "__kernel void operation(const float a,								\n"
						"	__private float * x, const long incx,							\n"
						"	__private float * y, const long incy)							\n"
						"{																	\n"
						"	int index = get_global_id(0);									\n"
						"	y[index * incy] = y[index * incy] + a * x[index * incx];		\n"
						"}																	\n";

	char const* daxpy = "__kernel void operation(const double a,							\n"
						"	__global double * x, const long incx,							\n"
						"	__global double * y, const long incy)							\n"
						"{																	\n"
						"	int index = get_global_id(0);									\n"
						"	y[index * incy] = y[index * incy] + a * x[index * incx];		\n"
						"}																	\n";
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

	while (incy <= 0 && incx <= 0)
	{
		std::cout << "IncY: ";
		std::cin >> incy;

		std::cout << "IncX: ";
		std::cin >> incx;
	}

	using op_type = float;
	std::vector<op_type> y(size, 1.);
	std::vector<op_type> x(size, 1.);
	op_type a{ 1.4 };

	my::_axpy(size, a, x.data(), incx, y.data(), incy);
	for (const auto& el : y)
	{
		std::cout << el << std::endl;
	}

	MY::DevWorker worker = MY::DevWorker();
	MY::GpuTask task = worker.createGpuTask(NVidia_device.c_str(), saxpy);
	if (!task.isTaskFailed())
	{
		std::cout << "GpuTask was created\n";

		int res = CL_SUCCESS;

		std::vector<cl_float> y_gpu(size, 1.);
		std::vector<cl_float> x_gpu(size, 1.);
		const cl_float a_gpu{ 1.4 };
		cl_long yBuffSize = y_gpu.size();
		cl_long xBuffSize = x_gpu.size();

		cl_mem yBuff = task.addBuffer<cl_float>(yBuffSize, CL_MEM_READ_WRITE, res);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in buffer creation process\n";
			return EXIT_FAILURE;
		}

		cl_mem xBuff = task.addBuffer<cl_float>(xBuffSize, CL_MEM_READ_ONLY, res);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in buffer creation process\n";
			return EXIT_FAILURE;
		}

		res = task.enqueueWriteBuffer<cl_float>(y_gpu.size(), y_gpu.data(), yBuff);
		if (res != CL_SUCCESS) return EXIT_FAILURE;
		res = task.enqueueWriteBuffer<cl_float>(x_gpu.size(), x_gpu.data(), xBuff);
		if (res != CL_SUCCESS) return EXIT_FAILURE;

		res = task.passParams(a_gpu, xBuff, incx, yBuff, incy);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in params passing process\n";
			std::cout << res << std::endl;
			return EXIT_FAILURE;
		}

		size_t localSize = 10;
		size_t globalSize = y_gpu.size();
		res = task.enqueueKernel(1, &localSize, &globalSize);
		if (res != CL_SUCCESS) return EXIT_FAILURE;

		res = task.enqueueReadBuffer<float>(y_gpu.size(), y_gpu.data(), yBuff);
		if (res != CL_SUCCESS) return EXIT_FAILURE;

		for (const auto& el : y_gpu)
		{
			std::cout << el << '\n';
		}

		clReleaseMemObject(yBuff);
		clReleaseMemObject(xBuff);
		return EXIT_SUCCESS;
	}
	else
	{
		std::cout << "GpuTask creation failed!\n";
	}
	return EXIT_FAILURE;
}