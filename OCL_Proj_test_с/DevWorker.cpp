#include "DevWorker.h"

namespace my
{
void DevWorker::initDevices()
{
	cl_uint platformCount{ 0 };
	clGetPlatformIDs(0, nullptr, &platformCount);
	platforms.resize(platformCount);
	clGetPlatformIDs(platformCount, platforms.data(), nullptr);

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

bool DevWorker::findDeviceByName(cl_platform_id& platformId, cl_device_id& deviceId, const char* _deviceName)
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
			if (fnd != nullptr)
			{
				std::cout << '\n' << deviceName << '\n';
				platformId = platform.first;
				deviceId = device;
				return true;
			}
		}
	}
	return false;
}

GpuTask DevWorker::createGpuTask(const char* _deviceName, const char* _sourceKernel)
{
	cl_device_id device;
	cl_platform_id platform;
	if (findDeviceByName(platform, device, _deviceName))
	{
		return GpuTask(device, _sourceKernel);
	}
	return GpuTask();
}

DevWorker::DevWorker()
{
	initDevices();
}
}