#pragma once
#include <vector>
#include <CL/cl.h>

std::vector<cl_int> matMultCpu(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeX, cl_int sizeY, cl_int sizeZ);
std::vector<cl_int> matMultCpuTransp(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeX, cl_int sizeY, cl_int sizeZ);
std::vector<cl_int> matMultCpuOMP(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeX, cl_int sizeY, cl_int sizeZ);
std::vector<cl_int> transpMatr(const std::vector<cl_int>& matrA, cl_int sizeX, cl_int sizeY);

std::vector<cl_int> matMultCpuTranspOMP(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeX, cl_int sizeY, cl_int sizeZ);
std::vector<cl_int> transpMatrOMP(const std::vector<cl_int>& matrA, cl_int sizeX, cl_int sizeY);

std::vector<cl_int> matMultCpuBlock(const std::vector<cl_int>& matrA, const std::vector<cl_int>& matrB, cl_int sizeX, cl_int sizeY, cl_int sizeZ);

std::vector<cl_int> matMultGpu(std::vector<cl_int>& matrA, std::vector<cl_int>& matrB, cl_int sizeZ, cl_int sizeY, cl_int sizeX,
	const char* device, bool useSharedMemory = false);

