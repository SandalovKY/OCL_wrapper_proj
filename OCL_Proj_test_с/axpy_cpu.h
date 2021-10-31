#include <cstdint>

namespace my
{
	template <typename fp_type>
	void _axpy(int64_t n, fp_type a, fp_type* x, int64_t incx, fp_type* y, int64_t incy)
	{
		incx = incx < 1 ? 1 : incx;
		incy = incy < 1 ? 1 : incy;
		if (n <= 0) return;
		size_t index = 0;
		while (index < n)
		{
			y[index * incy] += a * x[index * incx];
			++index;
		}
	}
}