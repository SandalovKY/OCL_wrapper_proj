#pragma once
#include <cstdint>
#include <vector>

namespace my
{
	template <typename fp_type>
	void _axpy(int64_t n, fp_type a, const std::vector<fp_type>& x, int64_t incx, std::vector<fp_type>& y, int64_t incy)
	{
		if (n <= 0 || incx <= 0 || incy <= 0) return;

		for (size_t index = 0; index < n; ++index)
		{
			if (index * incy >= y.size() ||
				index * incx >= x.size()) return;
			y[index * incy] += a * x[index * incx];
		}
	}

	void saxpy_omp(int64_t n, float a, const std::vector<float>& x, int64_t incx, std::vector<float>& y, int64_t incy)
	{
		if (n <= 0 || incx <= 0 || incy <= 0) return;

#pragma omp parallel for
		for (int64_t index = 0; index < n; ++index)
		{
			if (index * incy >= y.size() ||
				index * incx >= x.size()) break;
			y[index * incy] += a * x[index * incx];
		}
	}

	void daxpy_omp(int64_t n, double a, const std::vector<double>& x, int64_t incx, std::vector<double>& y, int64_t incy)
	{
		if (n <= 0 || incx <= 0 || incy <= 0) return;

#pragma omp parallel for
		for (int64_t index = 0; index < n; ++index)
		{
			if (index * incy >= y.size() ||
				index * incx >= x.size()) break;
			y[index * incy] += a * x[index * incx];
		}
	}
}