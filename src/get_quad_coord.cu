#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hemi.h"
#include "common.h"
#include <omp.h>

// assert() is only supported // for devices of compute capability 2.0 and higher 
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200) 
#undef  assert 
#define assert(arg) 
#endif

int get_cuda_device_count()
{
   int count = 1;
   cudaGetDeviceCount(&count);
   return count;
}

__constant__ double c_tau;
__constant__ double c_h;
__constant__ double c_tau_to_current_time_level;
__constant__ double c_lb;
__constant__ double c_rb;
__constant__ double c_ub;
__constant__ double c_bb;
__constant__ double c_tau_b;
__constant__ double c_pi_half;
__constant__ int c_x_length;
__constant__ int c_n;

__global__ void get_square_coord(double* first1, double* second1, double* third1,
	double* first2, double* second2, double* third2)
{
	for (int opt = blockIdx.x * blockDim.x + threadIdx.x; opt < c_n; opt += blockDim.x * gridDim.x)
	{
		int i = opt % c_x_length + 1;
		int j = opt / c_x_length + 1;
		double x, y;

		// A
		x = (c_h*(i - 1) + c_h*i) / 2.;
		y = (c_h*(j - 1) + c_h*j) / 2.;
		first1[2 * opt] = first2[2 * opt] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
		first1[2 * opt + 1] = first2[2 * opt + 1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_time_level * (y - c_ub) * (y - c_bb));

		// B
		x = (c_h*(i + 1) + c_h*i) / 2.;
		//	y = (c_h*(j - 1) + c_h*j) / 2.; // это значение совпадает со значением для предыдущей точки значит его можно не расчитывать
		second1[2 * opt] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
		second1[2 * opt + 1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_time_level * (y - c_ub) * (y - c_bb));

		// C
		//x = (a_x[i + 1] + a_x[i]) / 2.; // это значение совпадает со значением для предыдущей точки значит его можно не расчитывать
		y = (c_h*(j + 1) + c_h*j) / 2.;
		third1[2 * opt] = third2[2 * opt] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
		third1[2 * opt + 1] = third2[2 * opt + 1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_time_level * (y - c_ub) * (y - c_bb));

		// D 
		x = (c_h*(i - 1) + c_h*i) / 2.;
		//y = (a_y[j + 1] + a_y[j]) / 2.; // это значение совпадает со значением для предыдущей точки значит его можно не расчитывать
		second2[2 * opt] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
		second2[2 * opt + 1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_time_level * (y - c_ub) * (y - c_bb));
	}
}

__global__ void get_square_coord_first1_2(double *result)
{
	for (int opt = blockIdx.x * blockDim.x + threadIdx.x; opt < c_n; opt += blockDim.x * gridDim.x)
	{
		int i = opt % c_x_length + 1;
		int j = opt / c_x_length + 1;
		double x, y;

		// A
		x = (c_h*(i - 1) + c_h*i) / 2.;
		y = (c_h*(j - 1) + c_h*j) / 2.;
		result[2 * opt] =  x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
		result[2 * opt + 1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_time_level * (y - c_ub) * (y - c_bb));
	}
}

__global__ void get_square_coord_second1(double *result)
{
	for (int opt = blockIdx.x * blockDim.x + threadIdx.x; opt < c_n; opt += blockDim.x * gridDim.x)
	{
		int i = opt % c_x_length + 1;
		int j = opt / c_x_length + 1;

		double x, y;
// B
		x = (c_h*(i + 1) + c_h*i) / 2.;
		result[2 * opt] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
		result[2 * opt + 1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_time_level * (y - c_ub) * (y - c_bb));
		
	}
}

__global__ void get_square_coord_third1_2(double *result)
{
	for (int opt = blockIdx.x * blockDim.x + threadIdx.x; opt < c_n; opt += blockDim.x * gridDim.x)
	{
		int i = opt % c_x_length + 1;
		int j = opt / c_x_length + 1;

		double x, y;
// C
		//x = (a_x[i + 1] + a_x[i]) / 2.; // это значение совпадает со значением для предыдущей точки значит его можно не расчитывать
		y = (c_h*(j + 1) + c_h*j) / 2.;
		result[2 * opt]  = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
		result[2 * opt + 1]  = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_time_level * (y - c_ub) * (y - c_bb));
		
	}
}

__global__ void get_square_coord_second2(double *result)
{
	for (int opt = blockIdx.x * blockDim.x + threadIdx.x; opt < c_n; opt += blockDim.x * gridDim.x)
	{
		int i = opt % c_x_length + 1;
		int j = opt / c_x_length + 1;

		double x, y;
// D 
		x = (c_h*(i - 1) + c_h*i) / 2.;
		//y = (a_y[j + 1] + a_y[j]) / 2.; // это значение совпадает со значением для предыдущей точки значит его можно не расчитывать
		result[2 * opt] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
		result[2 * opt + 1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_time_level * (y - c_ub) * (y - c_bb));
			
	}
}

float get_quad_coord(TriangleResult* result, ComputeParameters* p)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	size_t size(0), n(0);
	int gridSize = 256;
	int blockSize =  512;
	double temp(0);
	n = p->get_inner_matrix_size();


	float elapsedTime;

#ifdef FLAGMAN

	// Start record
	cudaEventRecord(start, 0);

	cudaMemcpyToSymbol(c_tau, &p->tau, sizeof(double));
	cudaMemcpyToSymbol(c_lb, &p->lb, sizeof(double));
	cudaMemcpyToSymbol(c_rb, &p->rb, sizeof(double));
	cudaMemcpyToSymbol(c_bb, &p->bb, sizeof(double));
	cudaMemcpyToSymbol(c_ub, &p->ub, sizeof(double));
	cudaMemcpyToSymbol(c_n, &n, sizeof(int));
	cudaMemcpyToSymbol(c_x_length, &result->x_length, sizeof(int));
	temp = 1. / (result->x_length + 1);
	cudaMemcpyToSymbol(c_h, &temp, sizeof(double));

	temp = (1. + p->currentTimeLevel * p->tau) / 10.;
	cudaMemcpyToSymbol(c_tau_to_current_time_level, &temp, sizeof(double));

	temp = p->b * p->tau;
	cudaMemcpyToSymbol(c_tau_b, &temp, sizeof(double));

	temp = C_pi_device / 2.;
	cudaMemcpyToSymbol(c_pi_half, &temp, sizeof(double));

	size = 2 * sizeof(double)*n;
	
	
		int deviceCount = get_cuda_device_count();
	printf("Device Count = %d\n", deviceCount);
	
	 omp_set_num_threads(6);
	 #pragma omp parallel
	 {
	 
	 double *res= NULL;
	 
	 checkCuda(cudaMalloc((void**)&(res), size) );
	 int cpuId = omp_get_thread_num();
	 cudaSetDevice(cpuId);
	 
	// можно это ядро раскидать на карточки 
	// Вариант 1) На 1 карте считать first1, second1, third1, а на второй считать first2, second2, third2
	// Вариант 2) На 1 карте считать first1, на второй second1 и т. д.
	
	if (cpuId == 0 || cpuId == 1)
	{
	get_square_coord_first1_2 << <gridSize, blockSize >> >(res);
 }
 if (cpuId == 2)
	{
	get_square_coord_second1 << <gridSize, blockSize >> >(res);
 }
 if (cpuId == 3 || cpuId == 4)
 {
 get_square_coord_third1_2 << <gridSize, blockSize >> >(res);
 
 }
 
 if (cpuId == 5)
 {
 get_square_coord_second2 << <gridSize, blockSize >> >(res);
 
 }
	 
	 printf("Hi! I'm %d thread!\n", cpuId);
	 if (cpuId == 0)
	 {
	
	 cudaMemcpy(result->first1, res, size, cudaMemcpyDeviceToHost); 
	 }
	 else if (cpuId = 1)
	 
	{
	
	 cudaMemcpy(result->first2, res, size, cudaMemcpyDeviceToHost);	 
	 }
	 else if (cpuId = 2)
	 {
	 
	 cudaMemcpy(result->second1, res, size, cudaMemcpyDeviceToHost);
	 }
	 else if (cpuId = 3)
	 {
	 
	 cudaMemcpy(result->second2, res, size, cudaMemcpyDeviceToHost);
	 }
	 else if (cpuId = 4)
	 {
	 
	 cudaMemcpy(result->third1, res, size, cudaMemcpyDeviceToHost);
	 }
	 else if (cpuId == 5)
	 {
	 cudaMemcpy(result->third2, res, size, cudaMemcpyDeviceToHost);
	 }
 cudaFree(res);
	cudaDeviceReset();
	 
	 }
	  cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	
	cudaDeviceReset();
	return elapsedTime;
	#else 
	double *first1 = NULL, *second1 = NULL, *third1 = NULL, *first2 = NULL, *second2 = NULL, *third2 = NULL;

	// Start record
	cudaEventRecord(start, 0);

	cudaMemcpyToSymbol(c_tau, &p->tau, sizeof(double));
	cudaMemcpyToSymbol(c_lb, &p->lb, sizeof(double));
	cudaMemcpyToSymbol(c_rb, &p->rb, sizeof(double));
	cudaMemcpyToSymbol(c_bb, &p->bb, sizeof(double));
	cudaMemcpyToSymbol(c_ub, &p->ub, sizeof(double));
	cudaMemcpyToSymbol(c_n, &n, sizeof(int));
	cudaMemcpyToSymbol(c_x_length, &result->x_length, sizeof(int));
	temp = 1. / (result->x_length + 1);
	cudaMemcpyToSymbol(c_h, &temp, sizeof(double));

	temp = (1. + p->currentTimeLevel * p->tau) / 10.;
	cudaMemcpyToSymbol(c_tau_to_current_time_level, &temp, sizeof(double));

	temp = p->b * p->tau;
	cudaMemcpyToSymbol(c_tau_b, &temp, sizeof(double));

	temp = C_pi_device / 2.;
	cudaMemcpyToSymbol(c_pi_half, &temp, sizeof(double));

	size = 2 * sizeof(double)*n;
	checkCuda(cudaMalloc((void**)&(first1), size) );
	checkCuda(cudaMalloc((void**)&(second1), size));
	checkCuda(cudaMalloc((void**)&(third1), size) );
	checkCuda(cudaMalloc((void**)&(first2), size) );
	checkCuda(cudaMalloc((void**)&(second2), size));
	checkCuda(cudaMalloc((void**)&(third2), size) );  

	// можно это ядро раскидать на карточки 
	// Вариант 1) На 1 карте считать first1, second1, third1, а на второй считать first2, second2, third2
	// Вариант 2) На 1 карте считать first1, на второй second1 и т. д.
	get_square_coord << <gridSize, blockSize >> >(first1, second1, third1, first2, second2, third2);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(result->first1, first1, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(result->second1, second1, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(result->third1, third1, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(result->first2, first2, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(result->second2, second2, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(result->third2, third2, size, cudaMemcpyDeviceToHost);

	cudaFree(first1);
	cudaFree(second1);
	cudaFree(third1);
	cudaFree(first2);
	cudaFree(second2);
	cudaFree(third2);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();
	return elapsedTime;
	
	#endif
}