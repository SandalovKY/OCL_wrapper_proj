#pragma once

#include <CL/cl.h>
#include <vector>
#include <unordered_map>
#include <iostream>
namespace MY
{
	namespace
	{
		const uint8_t NAME_LENGTH{ 129 };
		const uint32_t O_SIZE{ 1000 };
	}

	class GpuTask
	{
	public:
		GpuTask() {}
		GpuTask(cl_device_id device, const char* _sourceKernel)
		{
			status = initProgram(device, _sourceKernel);
		}

		template <typename Arg>
		int passParam(int n, const Arg& arg)
		{
			return clSetKernelArg(kernel, n, sizeof(Arg), (void*)(&arg));
		}

		template <typename... Targs>
		int passParams(const Targs&... args)
		{
			return setArgs<0, Targs...>::set(kernel, args...);
		}
		int enqueueKernel(size_t numDims, size_t* localSize, size_t* global_size)
		{
			return clEnqueueNDRangeKernel(queue, kernel, numDims, NULL, global_size, localSize, 0, NULL, NULL);
		}
		bool isTaskFailed()
		{
			return status != CL_SUCCESS;
		}
		template <typename TYPE>
		cl_mem addBuffer(size_t size, int type, int& err)
		{
			return clCreateBuffer(context,
				type, sizeof(TYPE) * size, NULL, &err);
		}
		template <typename TYPE>
		int enqueueWriteBuffer(size_t size, TYPE* ptr, cl_mem memBuffer)
		{
			return clEnqueueWriteBuffer(queue, memBuffer, CL_TRUE, 0, sizeof(TYPE) * size, ptr, 0, NULL, NULL);
		}
		template <typename TYPE>
		int enqueueReadBuffer(size_t size, TYPE* ptr, cl_mem memBuffer)
		{
			return clEnqueueReadBuffer(queue, memBuffer, CL_TRUE, 0, sizeof(TYPE) * size, ptr, 0, NULL, NULL);
		}

		~GpuTask()
		{
			clReleaseProgram(program);
			clReleaseKernel(kernel);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
		}

	private:
		int initProgram(cl_device_id device, const char* _sourceKernel)
		{
			int err{};
			context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "context error!\n";
				return err;
			}
			queue = clCreateCommandQueue(context, device, 0, &err);
			if (err != CL_SUCCESS) {
				std::cout << "command queue error!\n";
				return err;
			}
			size_t srcLen = strlen(_sourceKernel);
			program = clCreateProgramWithSource(context, 1,
				(const char**)&_sourceKernel,
				&srcLen, &err);
			if (err != CL_SUCCESS) {
				std::cout << "program creation error!\n";
				return err;
			}
			err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
			if (err != CL_SUCCESS) {
				std::cout << "program building error!\n";
				return err;
			}
			kernel = clCreateKernel(program, "operation", &err);
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

		cl_context context{};
		cl_command_queue queue{};
		cl_program program{};
		cl_kernel kernel{};
		int status{-1};
	};
	
	class DevWorker
	{
	private:
		std::unordered_map<cl_platform_id, std::vector<cl_device_id>> allDevices;
		std::vector<cl_platform_id> platforms;

		void initDevices()
		{
			cl_uint platformCount{ 0 };
			clGetPlatformIDs(0, nullptr, &platformCount);
			platforms.resize(platformCount);
			allDevices.reserve(platformCount);
			clGetPlatformIDs(platformCount, platforms.data(), nullptr);

			for (const auto& platform : platforms)
			{
				char deviceName[NAME_LENGTH];
				size_t realSize{};
				clGetPlatformInfo(platform, CL_PLATFORM_NAME,
					NAME_LENGTH, deviceName, &realSize);
				deviceName[realSize] = '\0';
			}

			for (const auto& platform : platforms)
			{
				cl_uint numDevices{ 0 };
				clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
					0, nullptr, &numDevices);
				allDevices.emplace(platform, std::vector<cl_device_id>(numDevices));
				clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
					numDevices, allDevices.at(platform).data(), nullptr);
			}
		}

		bool findDeviceByName(cl_platform_id& platformId, cl_device_id& deviceId, const char* _deviceName)
		{
			for (const auto& platform : allDevices)
			{
				for (const auto& device : platform.second)
				{
					char deviceName[NAME_LENGTH];
					size_t realSize{};
					clGetDeviceInfo(device, CL_DEVICE_NAME,
						NAME_LENGTH, deviceName, &realSize);
					deviceName[realSize] = '\0';
					char* fnd = nullptr;
					fnd = strstr(deviceName, _deviceName);
					std::cout << deviceName << '\n';
					if (fnd != nullptr)
					{
						platformId = platform.first;
						deviceId = device;
						return true;
					}
				}
			}
			return false;
		}

	public:

		GpuTask createGpuTask(const char* _deviceName, const char* _sourceKernel)
		{
			cl_device_id device;
			cl_platform_id platform;
			if (findDeviceByName(platform, device, _deviceName))
			{
				return GpuTask(device, _sourceKernel);
			}
			return GpuTask();
		}

		DevWorker()
		{
			initDevices();
		}

		~DevWorker()
		{
		}
	};
}

