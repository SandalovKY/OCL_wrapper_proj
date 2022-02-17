#pragma once

#include <CL/cl.h>
#include <vector>
#include <map>
#include <iostream>
#include "GpuTask.h"

namespace my
{
	namespace
	{
		const uint8_t NAME_LENGTH{ 129 };
	}
	
	class DevWorker
	{
	private:
		std::map<cl_platform_id, std::vector<cl_device_id>> allDevices;
		std::vector<cl_platform_id> platforms;

		void initDevices();

		bool findDeviceByName(cl_platform_id& platformId, cl_device_id& deviceId, const char* _deviceName);

	public:

		GpuTask createGpuTask(const char* _deviceName, const char* _sourceKernel);

		DevWorker();
	};
}

