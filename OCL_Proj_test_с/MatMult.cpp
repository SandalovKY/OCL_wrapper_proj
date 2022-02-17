#include "MatMult.h"
#include "DevWorker.h"

namespace
{
	char const* matMultBase = 
		"__kernel void operation(const __global int * matrA,				\n"
		"	const __global int * matrB, __global int * resMatr,				\n"
		"	unsigned int Z, unsigned int Y, unsigned int X)					\n"
		"{																	\n"
		"	int z = get_global_id(0);										\n"
		"	int x = get_global_id(1);										\n"				
		"	int sum = 0;													\n"
		"	for (int y = 0; y < Y; ++y)										\n"
		"	{																\n"
		"		sum += matrA[z * Y + y] * matrB[y * X + x];					\n"
		"	}																\n"
		"	resMatr[z * X + x] = sum;										\n"
		"}																	\n";

	char const* matMulWithSharedMemory =
		"#define TILE_SIZE 16												\n"
		"__kernel void operation(const __global int * matrA,				\n"
		"	const __global int * matrB, __global int * resMatr,				\n"
		"	unsigned int Z, unsigned int Y, unsigned int X)					\n"
		"{																	\n"
		"	int z = get_global_id(0);										\n"
		"	int x = get_global_id(1);										\n"
		"	int lz = get_local_id(0);										\n"
		"	int lx = get_local_id(1);										\n"
		"																	\n"
		"	__local int tileA[TILE_SIZE][TILE_SIZE];						\n"
		"	__local int tileB[TILE_SIZE][TILE_SIZE];						\n"
		"																	\n"
		"	int sum = 0;													\n"
		"	for (int tileY = 0; tileY * TILE_SIZE < Y; ++tileY)				\n"
		"	{																\n"
		"		tileA[lz][lx] = matrA[z * Y + (tileY * TILE_SIZE + lx)];	\n"
		"		tileB[lz][lx] = matrB[(tileY * TILE_SIZE + lz) * X + x];	\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);								\n"
		"		for (int y = 0; y < TILE_SIZE; ++y)							\n"
		"		{															\n"
		"			sum += tileA[lz][y] * tileB[y][lx];						\n"
		"		}															\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);								\n"
		"	}																\n"
		"	resMatr[z * X + x] = sum;										\n"
		"}																	\n";
}

std::vector<cl_int> matMultGpu(std::vector<cl_int>& matrA, std::vector<cl_int>& matrB, cl_int sizeZ, cl_int sizeY, cl_int sizeX,
	const char* device, bool useSharedMemory)
{
	std::vector<cl_int> resMatr(sizeZ * sizeX, 0);
	my::DevWorker worker = my::DevWorker();

	my::GpuTask task = worker.createGpuTask(device, useSharedMemory ? matMulWithSharedMemory : matMultBase);
	if (!task.isTaskFailed())
	{
		int res = CL_SUCCESS;

		size_t localSize[]{ 16, 16 };
		size_t globalSize[]{ sizeX, sizeZ };
		size_t matrABuffer = matrA.size();
		size_t matrBBuffer = matrB.size();
		size_t resMatrBuffer = resMatr.size();

		double totalTime = omp_get_wtime();
		cl_mem matrABuff, matrBBuff, resMatrBuff;

		std::string AMD_device{ "gfx902" };

		if (AMD_device.find(device) != AMD_device.npos)
		{
			matrABuff = task.addBuffer<cl_int>(matrABuffer, CL_MEM_USE_HOST_PTR, res, matrA.data());
			if (res != CL_SUCCESS)
			{
				std::cout << "Problem in buffer creation process\n";
				return {};
			}

			matrBBuff = task.addBuffer<cl_int>(matrBBuffer, CL_MEM_USE_HOST_PTR, res, matrB.data());
			if (res != CL_SUCCESS)
			{
				std::cout << "Problem in buffer creation process\n";
				return {};
			}

			resMatrBuff = task.addBuffer<cl_int>(resMatrBuffer, CL_MEM_USE_HOST_PTR, res, resMatr.data());
			if (res != CL_SUCCESS)
			{
				std::cout << "Problem in buffer creation process\n";
				return {};
			}
		}
		else
		{
			matrABuff = task.addBuffer<cl_int>(matrABuffer, CL_MEM_READ_ONLY, res);
			if (res != CL_SUCCESS)
			{
				std::cout << "Problem in buffer1 creation process\n";
				return {};
			}

			matrBBuff = task.addBuffer<cl_int>(matrBBuffer, CL_MEM_READ_ONLY, res);
			if (res != CL_SUCCESS)
			{
				std::cout << "Problem in buffer2 creation process\n";
				return {};
			}

			resMatrBuff = task.addBuffer<cl_int>(resMatrBuffer, CL_MEM_WRITE_ONLY, res);
			if (res != CL_SUCCESS)
			{
				std::cout << "Problem in buffer3 creation process\n";
				return {};
			}

			res = task.enqueueWriteBuffer<cl_int>(matrABuffer, matrA.data(), matrABuff);
			if (res != CL_SUCCESS)
			{
				std::cout << res << '\n';
				std::cout << "Problem in write buffer enqueue\n";
				return {};
			}
			res = task.enqueueWriteBuffer<cl_int>(matrBBuffer, matrB.data(), matrBBuff);
			if (res != CL_SUCCESS)
			{
				std::cout << res << '\n';
				std::cout << "Problem in write buffer enqueue\n";
				return {};
			}
		}

		res = task.passParams(matrABuff, matrBBuff, resMatrBuff, sizeZ, sizeY, sizeX);
		if (res != CL_SUCCESS)
		{
			std::cout << "Problem in params passing process\n";
			std::cout << res << std::endl;
			return {};
		}

		double kernelTime{};

		res = task.enqueueKernel(2, localSize, globalSize, &kernelTime);
		if (res != CL_SUCCESS)
		{
			std::cout << "With enqueue task proc problems\n";
			return {};
		}
		if (AMD_device.find(device) == AMD_device.npos)
		{
			std::cout << "HERE\n";
			res = task.enqueueReadBuffer<cl_int>(resMatrBuffer, resMatr.data(), resMatrBuff);
			if (res != CL_SUCCESS)
			{
				std::cout << "Problem in read buffer enqueue\n";
				return {};
			}
		}
		totalTime = omp_get_wtime() - totalTime;
		std::cout << "Kernel time on GPU: " << kernelTime << '\n';
		std::cout << "Total time on GPU: " << totalTime << '\n';
		clReleaseMemObject(matrABuff);
		clReleaseMemObject(matrBBuff);
		clReleaseMemObject(resMatrBuff);
		return resMatr;
	}
	else
	{
		std::cout << "GpuTask creation failed!\n";
		return {};
	}
}

std::vector<cl_int> matMultCpu(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeZ, cl_int sizeY, cl_int sizeX)
{
	std::vector<cl_int> resMatr(sizeX * sizeZ);
	for (int64_t z = 0; z < sizeZ; ++z)
	{
		for (int64_t x = 0; x < sizeX; ++x)
		{
			cl_int tmp = 0;
			for (int64_t y = 0; y < sizeY; ++y)
			{
				tmp += matrA[z * sizeY + y] * matrB[y * sizeX + x];
			}
			resMatr[z * sizeX + x] = tmp;
		}
	}
	return resMatr;
}

std::vector<cl_int> matMultCpuTransp(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeZ, cl_int sizeY, cl_int sizeX)
{
	std::vector<cl_int> resMatr(sizeX * sizeZ);
	for (int64_t z = 0; z < sizeZ; ++z)
	{
		for (int64_t x = 0; x < sizeX; ++x)
		{
			cl_int tmp = 0;
			for (int64_t y = 0; y < sizeY; ++y)
			{
				tmp += matrA[z * sizeY + y] * matrB[x * sizeY + y];
			}
			resMatr[z * sizeX + x] = tmp;
		}
	}
	return resMatr;
}

std::vector<cl_int> matMultCpuBlock(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeZ, cl_int sizeY, cl_int sizeX)
{
	std::vector<cl_int> resMatr(sizeX * sizeZ);

	int64_t yBlock = 16;
	int64_t xBlock = 16;
	int64_t zBlock = 16;

	for (int64_t z = 0; z < sizeZ; z += zBlock)
	{
		for (int64_t x = 0; x < sizeX; x += xBlock)
		{
			for (int64_t y = 0; y < sizeY; y += yBlock)
			{
				for (int64_t zb = z; zb < z + zBlock; ++zb)
				{
					for (int64_t xb = x; xb < x + xBlock; ++xb)
					{
						cl_int tmp = 0;
						for (int64_t yb = y; yb < y + yBlock; ++yb)
						{
							tmp += matrA[zb * sizeY + yb] * matrB[yb * sizeX + xb];
						}
						resMatr[zb * sizeX + xb] += tmp;
					}
				}
			}
		}
	}
	return resMatr;
}

std::vector<cl_int> transpMatr(const std::vector<cl_int>& matrA, cl_int sizeX, cl_int sizeY)
{
	std::vector<cl_int> resMatr(sizeX * sizeY);
	for (int64_t y = 0; y < sizeY; ++y)
	{
		for (int64_t x = 0; x < sizeX; ++x)
		{
			resMatr[x * sizeY + y] = matrA[y * sizeX + x];
		}
	}
	return resMatr;
}

std::vector<cl_int> matMultCpuOMP(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeZ, cl_int sizeY, cl_int sizeX)
{
	std::vector<cl_int> resMatr(sizeX * sizeZ);
#pragma omp parallel for num_threads(8)
	for (int64_t z = 0; z < sizeZ; ++z)
	{
		for (int64_t x = 0; x < sizeX; ++x)
		{
			cl_int tmp = 0;
			for (int64_t y = 0; y < sizeY; ++y)
			{
				tmp += matrA[z * sizeY + y] * matrB[y * sizeX + x];
			}
			resMatr[z * sizeX + x] = tmp;
		}
	}
	return resMatr;
}

std::vector<cl_int> matMultCpuTranspOMP(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeZ, cl_int sizeY, cl_int sizeX)
{
	std::vector<cl_int> resMatr(sizeX * sizeZ);
#pragma omp parallel for num_threads(8)
	for (int64_t z = 0; z < sizeZ; ++z)
	{
		for (int64_t x = 0; x < sizeX; ++x)
		{
			cl_int tmp = 0;
			for (int64_t y = 0; y < sizeY; ++y)
			{
				tmp += matrA[z * sizeY + y] * matrB[x * sizeY + y];
			}
			resMatr[z * sizeX + x] = tmp;
		}
	}
	return resMatr;
}

std::vector<cl_int> transpMatrOMP(const std::vector<cl_int>& matrA, cl_int sizeX, cl_int sizeY)
{
	std::vector<cl_int> resMatr(sizeX * sizeY);
#pragma omp parallel for num_threads(8)
	for (int64_t y = 0; y < sizeY; ++y)
	{
		for (int64_t x = 0; x < sizeX; ++x)
		{
			resMatr[x * sizeY + y] = matrA[y * sizeX + x];
		}
	}
	return resMatr;
}
