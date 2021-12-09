#pragma once
#include <cstdint>
#include <vector>

#include "DevWorker.h"

namespace my
{
namespace
{
	char const* saxpy = "__kernel void operation(long n, float a,			\n"
		"	const __global float * x, long incx, long xSize,				\n"
		"	__global float * y, long incy, long ySize)						\n"
		"{																	\n"
		"	int index = get_global_id(0);									\n"
		"	if (index * incx >= xSize || index * incy >= ySize)	return;		\n"
		"	if (index < n)													\n"
		"		y[index * incy] = y[index * incy] + a * x[index * incx]; 	\n"
		"}																	\n";

	char const* daxpy = "__kernel void operation(long n, double a,			\n"
		"	const __global double * x, long incx, long xSize,				\n"
		"	__global double * y, long incy, long ySize)						\n"
		"{																	\n"
		"	int index = get_global_id(0);									\n"
		"	if (index * incx >= xSize || index * incy >= ySize)	return;		\n"
		"	if (index < n)													\n"
		"		y[index * incy] = y[index * incy] + a * x[index * incx]; 	\n"
		"}																	\n";
}

int saxpy_gpu(size_t size, cl_float a_gpu, const std::vector<cl_float>& x_gpu, cl_long incx, std::vector<cl_float>& y_gpu, cl_long incy, const char* _deviceName)
{
	if (size <= 0 || incx <= 0 || incy <= 0) return EXIT_FAILURE;
	my::DevWorker worker = my::DevWorker();
	my::GpuTask task = worker.createGpuTask(_deviceName, saxpy);
	if (!task.isTaskFailed())
	{

		int res = CL_SUCCESS;

		size_t localSize{};
		size_t globalSize{};
		task.getDecomposition(&localSize, &globalSize, &size);
		size_t yBuffSize = y_gpu.size();
		size_t xBuffSize = x_gpu.size();

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

		double totalTime = omp_get_wtime();
		res = task.enqueueWriteBuffer<cl_float>(yBuffSize, y_gpu.data(), yBuff);
		if (res != CL_SUCCESS)
		{
			std::cout << res << '\n';
			std::cout << "Problem in write buffer enqueue\n";
			return EXIT_FAILURE;
		}
		res = task.enqueueWriteBuffer<cl_float>(xBuffSize, x_gpu.data(), xBuff);
		if (res != CL_SUCCESS)
		{
			std::cout << res << '\n';
			std::cout << "Problem in write buffer enqueue\n";
			return EXIT_FAILURE;
		}

		res = task.passParams(static_cast<cl_long>(size), a_gpu, xBuff, incx, xBuffSize, yBuff, incy, yBuffSize);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in params passing process\n";
			std::cout << res << std::endl;
			return EXIT_FAILURE;
		}

		double kernelTime{};
		res = task.enqueueKernel(1, &localSize, &globalSize, &kernelTime);
		if (res != CL_SUCCESS)
		{
			std::cout << "With enqueue task proc problems\n";
			return EXIT_FAILURE;
		}
		res = task.enqueueReadBuffer<float>(y_gpu.size(), y_gpu.data(), yBuff, CL_FALSE);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in read buffer enqueue\n";
			return EXIT_FAILURE;
		}
		totalTime = omp_get_wtime() - totalTime;
		std::cout << "Kernel time on GPU: " << kernelTime << '\n';
		std::cout << "Total time on GPU: " << totalTime << '\n';
		clReleaseMemObject(yBuff);
		clReleaseMemObject(xBuff);
		return EXIT_SUCCESS;
	}
	else
	{
		std::cout << "GpuTask creation failed!\n";
		return EXIT_FAILURE;
	}
}

int daxpy_gpu(size_t size, cl_double a_gpu, const std::vector<cl_double>& x_gpu, cl_long incx, std::vector<cl_double>& y_gpu, cl_long incy, const char* _deviceName)
{
	if (size <= 0 || incx <= 0 || incy <= 0) return EXIT_FAILURE;
	my::DevWorker worker = my::DevWorker();
	my::GpuTask task = worker.createGpuTask(_deviceName, daxpy);
	if (!task.isTaskFailed())
	{

		int res = CL_SUCCESS;

		size_t localSize{};
		size_t globalSize{};
		task.getDecomposition(&localSize, &globalSize, &size);
		size_t yBuffSize = y_gpu.size();
		size_t xBuffSize = x_gpu.size();

		cl_mem yBuff = task.addBuffer<cl_double>(yBuffSize, CL_MEM_READ_WRITE, res);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in buffer creation process\n";
			return EXIT_FAILURE;
		}

		cl_mem xBuff = task.addBuffer<cl_double>(xBuffSize, CL_MEM_READ_ONLY, res);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in buffer creation process\n";
			return EXIT_FAILURE;
		}

		double totalTime = omp_get_wtime();
		res = task.enqueueWriteBuffer<cl_double>(yBuffSize, y_gpu.data(), yBuff);
		if (res != CL_SUCCESS)
		{
			std::cout << res << '\n';
			std::cout << "Problem in write buffer enqueue\n";
			return EXIT_FAILURE;
		}
		res = task.enqueueWriteBuffer<cl_double>(xBuffSize, x_gpu.data(), xBuff);
		if (res != CL_SUCCESS)
		{
			std::cout << res << '\n';
			std::cout << "Problem in write buffer enqueue\n";
			return EXIT_FAILURE;
		}

		res = task.passParams(static_cast<cl_long>(size), a_gpu, xBuff, incx, xBuffSize, yBuff, incy, yBuffSize);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in params passing process\n";
			std::cout << res << std::endl;
			return EXIT_FAILURE;
		}

		double kernelTime{};
		res = task.enqueueKernel(1, &localSize, &globalSize, &kernelTime);
		if (res != CL_SUCCESS)
		{
			std::cout << "With enqueue task proc problems\n";
			return EXIT_FAILURE;
		}
		res = task.enqueueReadBuffer<cl_double>(y_gpu.size(), y_gpu.data(), yBuff);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in read buffer enqueue\n";
			return EXIT_FAILURE;
		}
		totalTime = omp_get_wtime() - totalTime;
		std::cout << "Kernel time on GPU: " << kernelTime << '\n';
		std::cout << "Total time on GPU: " << totalTime << '\n';
		clReleaseMemObject(yBuff);
		clReleaseMemObject(xBuff);
		return EXIT_SUCCESS;
	}
	else
	{
		std::cout << "GpuTask creation failed!\n";
		return EXIT_FAILURE;
	}
}
}