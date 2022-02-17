#pragma once
#include <CL/cl.h>
#include <iostream>
#include <omp.h>

namespace my
{

class GpuTask
{
public:
	GpuTask() = default;
	GpuTask(cl_device_id device, const char* _sourceKernel) : m_device(device)
	{
		status = initProgram(device, _sourceKernel);
	}

	template <typename Arg>
	int passParam(int n, const Arg& arg)
	{
		return clSetKernelArg(m_kernel, n, sizeof(Arg), (void*)(&arg));
	}

	template <typename... Targs>
	int passParams(const Targs&... args)
	{
		return setArgs<0, Targs...>::set(m_kernel, args...);
	}
	int enqueueKernel(size_t numDims, size_t* localSize, size_t* global_size, double* totalTime)
	{
		cl_event event{};
		*totalTime = omp_get_wtime();
		int retCode = clEnqueueNDRangeKernel(m_queue, m_kernel, numDims, NULL, global_size, localSize, 0, NULL, &event);
		clWaitForEvents(1, &event);
		*totalTime = omp_get_wtime() - *totalTime;
		clReleaseEvent(event);
		return retCode;
	}
	bool isTaskFailed()
	{
		return status != CL_SUCCESS;
	}
	template <typename TYPE>
	cl_mem addBuffer(size_t size, int type, int& err, TYPE* pointer = NULL)
	{
		return clCreateBuffer(m_context,
			type, sizeof(TYPE) * size, pointer, &err);
	}
	template <typename TYPE>
	int enqueueWriteBuffer(size_t size, TYPE* ptr, cl_mem memBuffer, size_t blockingWrite = CL_TRUE)
	{
		return clEnqueueWriteBuffer(m_queue, memBuffer, blockingWrite, 0, sizeof(TYPE) * size, ptr, 0, NULL, NULL);
	}
	template <typename TYPE>
	int enqueueWriteBuffer(size_t size, const TYPE* ptr, cl_mem memBuffer, size_t blockingWrite = CL_TRUE)
	{
		return clEnqueueWriteBuffer(m_queue, memBuffer, blockingWrite, 0, sizeof(TYPE) * size, ptr, 0, NULL, NULL);
	}
	template <typename TYPE>
	int enqueueReadBuffer(size_t size, TYPE* ptr, cl_mem memBuffer, size_t blockingRead = CL_TRUE)
	{
		return clEnqueueReadBuffer(m_queue, memBuffer, blockingRead, 0, sizeof(TYPE) * size, ptr, 0, NULL, NULL);
	}

	// add n-dim
	void getDecomposition(size_t* localSize, size_t* globalSize, const size_t* worksize)
	{
		*localSize = 128;
		cl_uint computeUnits{};
		int retCode = clGetDeviceInfo(m_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits, NULL);
		if (retCode == CL_SUCCESS)
		{
			cl_uint computeUnitsInWork = *worksize / *localSize;
			if (*worksize % *localSize)
			{
				++computeUnitsInWork;
			}
			if (computeUnitsInWork < computeUnits)
			{
				*localSize = 64;
			}
		}
		else
		{
			std::cout << "Error while finding optimal decomposition: " << retCode << std::endl;
		}

		if (*worksize % *localSize)
		{
			*globalSize = ((*worksize / *localSize) + 1) * *localSize;
		}
		else *globalSize = *worksize;
		return;
	}

	~GpuTask()
	{
		clReleaseProgram(m_program);
		clReleaseKernel(m_kernel);
		clReleaseCommandQueue(m_queue);
		clReleaseContext(m_context);
		clReleaseDevice(m_device);
	}
private:
	int initProgram(cl_device_id device, const char* _sourceKernel)
	{
		int err{};
		m_context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "context error!\n";
			return err;
		}
		m_queue = clCreateCommandQueueWithProperties(m_context, device, 0, &err);
		if (err != CL_SUCCESS) {
			std::cout << "command queue error!\n";
			return err;
		}
		size_t srcLen = strlen(_sourceKernel);
		m_program = clCreateProgramWithSource(m_context, 1,
			(const char**)&_sourceKernel,
			&srcLen, &err);
		if (err != CL_SUCCESS) {
			std::cout << "program creation error!\n";
			return err;
		}
		err = clBuildProgram(m_program, 1, &device, NULL, NULL, NULL);
		if (err != CL_SUCCESS) {
			std::cout << "program building error!\n";
			return err;
		}
		m_kernel = clCreateKernel(m_program, "operation", &err);
		if (err != CL_SUCCESS) {
			std::cout << "kernel error!\n";
			return err;
		}
		return CL_SUCCESS;
	}

	template <int Ind, typename... Args>
	struct setArgs;

	template<int Ind, typename Head, typename... Args>
	struct setArgs<Ind, Head, Args...>
	{
		static int set(const cl_kernel& kernel, const Head& head, const Args&... args)
		{
			int err = clSetKernelArg(kernel, Ind, sizeof(Head), (void*)(&head));
			return (err == CL_SUCCESS) ? setArgs<Ind + 1, Args...>::set(kernel, args...) : err;
		}
	};

	template<int Ind>
	struct setArgs<Ind>
	{
		static int set(const cl_kernel& kernel)
		{
			return CL_SUCCESS;
		}
	};

	cl_context m_context{};
	cl_command_queue m_queue{};
	cl_program m_program{};
	cl_kernel m_kernel{};
	cl_device_id m_device{};
	int status{ -1 };
};
}
