
#include "timer.h"
#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"
#include "model_data_provider.h"
#include <iostream>
#include <fstream>
#include <assert.h>     /* assert */
#ifndef COMMON_H_INCLUDED
#include "common.h"
#endif
#define FULL_TEST true
using namespace std;

class TestBase : public testing::Test
{
	protected:
		double _accuracy;

		ModelDataProvider _modelDataProvider;

		TestBase()
		{
			_accuracy = 1.0e-8;
			initCompOfGlVar();
			_modelDataProvider = ModelDataProvider();
		}

		virtual ~TestBase()
		{
			// You can do clean-up work that doesn't throw exceptions here.
			memClean();
		}

		// If the constructor and destructor are not enough for setting up
		// and cleaning up each test, you can define the following methods:

		virtual void SetUp()
		{
			// Code here will be called immediately after the constructor (right
			// before each test).
		}

		virtual void TearDown()
		{
			// Code here will be called immediately after each test (right
			// before the destructor).
		}

		int GetSize()
		{
			return C_numOfOXSt * C_numOfOYSt;
		}

		void print_matrix(int n, int m, double *a, int precision = 8)
		{
			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < m; ++j)
				{
					int k = i * n + j;
					switch (precision)
					{
						case 1:
							printf("%.1f ", a[k]);
							break;
						case 2:
							printf("%.2f ", a[k]);
							break;
						case 3:
							printf("%.3f ", a[k]);
							break;
						case 4:
							printf("%.4f ", a[k]);
							break;
						case 5:
							printf("%.5f ", a[k]);
							break;
						case 6:
							printf("%.6f ", a[k]);
							break;
						case 7:
							printf("%.7f ", a[k]);
							break;
						case 8:
							printf("%.8f ", a[k]);
							break;
					}
				}
				printf("\n");
			}
		}

		void print_matrix_to_file(int n, int m, double *data, std::string file_name, int precision = 8)
		{

			FILE * pFile;


			pFile = fopen (file_name.c_str(), "w");





			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < m; ++j)
				{
					int k = i * n + j;
					switch (precision)
					{

						case 8:
							fprintf (pFile, "%le ", data[k]);
							break;
					}
				} 
				fprintf (pFile, "\n ");
			}
			fclose (pFile);
		}
};

class cputest : public TestBase
{
	protected:

		cputest()
		{
		}

		virtual ~cputest()
		{
		}

		double *GetCpuToLevel(int level)
		{
			return solve_cpu_test(C_par_a, C_par_b, C_lbDom, C_rbDom, C_bbDom,
					C_ubDom, C_tau, C_numOfTSt, masOX, C_numOfOXSt, masOY,
					C_numOfOYSt, level, false);
		}

};

TEST_F(cputest, CpuTestModel11)
{
	double *data = _modelDataProvider.GetModelData(Model11);
	double *result = GetCpuToLevel(0);

	for (int i = 0; i < GetSize(); i++)
	{
		ASSERT_TRUE(abs(data[i] - result[i]) <= _accuracy) << data[i] << " " << result[i];

	}
}

TEST_F(cputest, CpuTestModel21)
{
	double *data = _modelDataProvider.GetModelData(Model21);
	double *result = GetCpuToLevel(1);

	for (int i = 0; i < GetSize(); i++)
	{
		EXPECT_TRUE(data[i] - result[i] <= _accuracy);
	}
}

TEST_F(cputest, CpuTestModel41)
{
	if (FULL_TEST)
	{
		double *data = _modelDataProvider.GetModelData(Model41);
		double *result = GetCpuToLevel(2);

		for (int i = 0; i < GetSize(); i++)
		{
			EXPECT_TRUE(data[i] - result[i] <= _accuracy);
		}
	}
}

TEST_F(cputest, CpuTestModel81)
{
	if (FULL_TEST)
	{
		double *data = _modelDataProvider.GetModelData(Model81);
		double *result = GetCpuToLevel(3);

		for (int i = 0; i < GetSize(); i++)
		{
			ASSERT_TRUE(data[i] - result[i] <= _accuracy);
		}
	}
}

TEST_F(cputest, CpuTestModel161)
{
	if (FULL_TEST)
	{
		double *data = _modelDataProvider.GetModelData(Model161);
		double *result = GetCpuToLevel(4);

		for (int i = 0; i < GetSize(); i++)
		{
			ASSERT_TRUE(data[i] - result[i] <= _accuracy);
		}
	}
}

TEST_F(cputest, CpuTestModel321)
{
	if (FULL_TEST)
	{
		double *data = _modelDataProvider.GetModelData(Model321);
		double *result = GetCpuToLevel(5);

		for (int i = 0; i < GetSize(); i++)
		{
			ASSERT_TRUE(data[i] - result[i] <= _accuracy);
		}
	}
}

TEST_F(cputest, CpuTestModel641)
{
	if (FULL_TEST)
	{
		double *data = _modelDataProvider.GetModelData(Model641);
		double *result = GetCpuToLevel(6);

		for (int i = 0; i < GetSize(); i++)
		{
			ASSERT_TRUE(data[i] - result[i] <= _accuracy);
		}
	}
}

TEST_F(cputest, CpuTestModel1281)
{
	if (FULL_TEST)
	{
		double *data = _modelDataProvider.GetModelData(Model1281);
		double *result = GetCpuToLevel(7);

		for (int i = 0; i < GetSize(); i++)
		{
			ASSERT_TRUE(data[i] - result[i] <= _accuracy);
		}
	}
}

class gputest : public TestBase
{
	protected:

		ModelDataProvider _modelDataProvider;
		gputest()
		{
			_modelDataProvider = ModelDataProvider();
		}

		virtual ~gputest()
		{
		}

		double *GetCpuToLevel(int level, bool isComputeDiff = false)
		{

			return solve_cpu_test(C_par_a, C_par_b, C_lbDom, C_rbDom, C_bbDom,
					C_ubDom, C_tau, C_numOfTSt, masOX, C_numOfOXSt, masOY,
					C_numOfOYSt, level, isComputeDiff);
		}


		double *GetCpuToLevel1TL(int level, bool isComputeDiff)
		{
			return solve_cpu_test(C_par_a, C_par_b, C_lbDom, C_rbDom, C_bbDom,
					C_ubDom, C_tau, 1, masOX, C_numOfOXSt, masOY,
					C_numOfOYSt, level, isComputeDiff);
		}
};

TEST_F(gputest, main_test)
{
	const int finishLevel = 8;
	const int startLevel = 0;
	const bool isComputeDiff = true;
	const bool isOneTl = false;

	for (int level = startLevel; level < finishLevel; ++level)
	{

		double *data = NULL;

		ComputeParameters *p = new ComputeParameters(level, true, isComputeDiff);
		ASSERT_TRUE(p->result != NULL);
		std::cout << *p << std::endl;

		float gpu_time = solve_at_gpu(p, isOneTl, isComputeDiff);
		ASSERT_TRUE(gpu_time != -1);
		if (isOneTl)
		{		data = GetCpuToLevel1TL(level, isComputeDiff); }
		else
		{  data = GetCpuToLevel(level, isComputeDiff); }


		//	        data = _modelDataProvider.GetModelData(level);


		printf("%s\n", "Start testing...");

		for (int i = 0; i < p->get_real_matrix_size(); i++)
		{
			ASSERT_NEAR(data[i], p->result[i], __FLT_EPSILON__) << "i = " <<  i << std::endl;
		}

		std::ostringstream oss; 

		char *s = "diff_gpu_"; 
		oss << s << p->t_count << ".bin"; 

		std::string name(oss.str());

		if (isComputeDiff)
		{
			print_matrix_to_file(p->get_real_x_size(), p->get_real_y_size(), p->diff, name); 
		}


		delete p;
		delete[] data;
	}
}


TEST_F(gputest, main_test_te)
{
	const int finishLevel = 9;
	const int startLevel = 0;
	const double error = 1.0e-8;
	double time_cpu = -1;

	for (int level = startLevel; level < finishLevel; ++level)
	{
		std::cout << "level = " << level << std::endl;
		ComputeParameters *p = new ComputeParameters(level, true);
		ASSERT_TRUE(p->result != NULL);

		printf("Start GPU\n");
		float time_gpu = solve_at_gpu(p, false);
		printf("End GPU\n");

		printf("Start CPU\n");
		StartTimer();
		double *data = GetCpuToLevel(level);
		time_cpu = GetTimer();
		printf("End CPU\n");

		printf("CPU time is = %f\n", time_cpu);
		printf("GPU time is = %f\n", time_gpu);
		printf("CPU/GPU = %f\n", time_cpu / time_gpu);

		printf("%s\n", "Start checking...");

		for (int i = 0; i < p->get_real_matrix_size(); i++)
		{
			ASSERT_DOUBLE_EQ(data[i], p->result[i]);
		}

		delete p;
		delete[] data;
	}
}



TEST_F(gputest, main_test_1tl_boundaries)
{
	const int finishLevel = 1;
	const int startLevel = 0;
	const double error = 1.0e-8;

	ComputeParameters *p = new ComputeParameters(0, true);
	ASSERT_TRUE(p->result != NULL);
	float gpu_time = solve_at_gpu(p, true);
	ASSERT_TRUE(gpu_time != -1);
	double *data = _modelDataProvider.GetModelData1tl(0);
	/*print_matrix(p->get_real_x_size(), p->get_real_y_size(), data);
	  printf("%s\n", "");
	  print_matrix(p->get_real_x_size(), p->get_real_y_size(), p->result);*/
	printf("%s\n", "Start testing...");
	for (int i = 0; i < p->get_real_matrix_size(); i++)
	{
		int n = i % p->get_real_x_size();
		int m = i / p->get_real_y_size();

		// расчет границы
		if (m == 0 || n == 0 || m == p->get_real_y_size() - 1 || n == p->get_real_x_size() - 1)
		{
			ASSERT_TRUE(fabs(data[i] - p->result[i]) <= error) << i << " " << data[i] << " " << p->result[i] << std::endl;
		}
	}

	delete p;
	delete[] data;
}

TEST_F(gputest, main_test_1tl_inner)
{
	const int finishLevel = 1;
	const int startLevel = 0;
	const double error = 1.0e-8;

	ComputeParameters *p = new ComputeParameters(0, true);
	ASSERT_TRUE(p->result != NULL);
	float gpu_time = solve_at_gpu(p, true);
	ASSERT_TRUE(gpu_time != -1);
	double *data = _modelDataProvider.GetModelData1tl(0);
	//double* data = GetCpuToLevel(0);
	printf("%s\n", "cpu");
	print_matrix(p->get_real_x_size(), p->get_real_y_size(), data, 5);
	printf("%s\n", "gpu");
	print_matrix(p->get_real_x_size(), p->get_real_y_size(), p->result, 5);
	printf("%s\n", "Start testing...");
	for (int i = 0; i < p->get_real_matrix_size(); i++)
	{
		int n = i % p->get_real_x_size();
		int m = i / p->get_real_y_size();

		// расчет границы
		if (m == 0 || n == 0 || m == p->get_real_y_size() - 1 || n == p->get_real_x_size() - 1)
		{
			continue;
		}
		ASSERT_TRUE(fabs(data[i] - p->result[i]) <= error) << i << " " << data[i] << " " << p->result[i] << std::endl;
	}

	delete p;
	delete[] data;
}

TEST_F(gputest, gen_1tl)
{
	const int finishLevel = 1;
	const double error = 1.0e-8;
	double *tl1 = GetCpuToLevel(0);
	delete[] tl1;
	//print_matrix(11, 11, tl1);
}

// This is the test checks that gpu and cpu results are equal for first 
// time layer
TEST_F(gputest, gen_1tl_7)
{
	const int finishLevel = 1;
	const double error = 1.0e-8;
	std::cout << "level = " << 7 << std::endl;
	ComputeParameters *p = new ComputeParameters(7, true);
	double *data = GetCpuToLevel(7);
	print_matrix_to_file(p->get_real_x_size(), p->get_real_y_size(), data, "1281_1281_6400_cpu_model_1tl.txt");

	ASSERT_TRUE(p->result != NULL);

	printf("Start GPU\n");
	//float time_gpu = solve_at_gpu(p, true);
	printf("End GPU\n");
	//for (int i = 0; i < p->get_real_matrix_size(); i++)
	{
		//        ASSERT_TRUE(fabs(data[i] - p->result[i]) <= error) << i << " " << data[i] << " " << p->result[i] << std::endl;
	}
	delete[] data;
	delete p; 
}


TEST_F(gputest, get_1281_result)
{
	const int finishLevel = 1;
	const double error = 1.0e-8;
	std::cout << "level = " << 7 << std::endl;
	ComputeParameters *p = new ComputeParameters(7, true);
	double *data = GetCpuToLevel(7);
	print_matrix_to_file(p->get_real_x_size(), p->get_real_y_size(), data, "1281_1281_6400_cpu_model.txt");
	delete[] data;
	delete p; 
}


TEST_F(gputest, get_error_for_1281_1tl)
{
	const int level = 7;
	const double error = 1.0e-15;
	ComputeParameters *p = new ComputeParameters(level, true);
	ASSERT_TRUE(p->result != NULL);
	float gpu_time = solve_at_gpu(p, true);
	//    double *data = _modelDataProvider.GetModelData1tl(level);
	double *data = GetCpuToLevel(level);
	//print_matrix(p->get_real_x_size(), p->get_real_y_size(), data);
	//printf("%d\n", p->get_real_matrix_size());
	double *diff = new double[p->get_real_matrix_size()];
	for (int i = 0; i < p->get_real_matrix_size(); ++i)
	{
		diff[i] = fabs(p->result[i] - data[i]);
		ASSERT_TRUE(fabs(data[i] - p->result[i]) <= error) << i << " " << data[i] << " " << p->result[i] << std::endl;
	}
	//  print_matrix(p->get_real_x_size(), p->get_real_y_size(), diff);
	print_matrix_to_file(p->get_real_x_size(), p->get_real_y_size(), diff, "diff_1281.txt");

	delete p;
	delete[] diff;
}

TEST_F(gputest, gen_2tl)
{
	const int finishLevel = 1;
	const int startLevel = 0;
	const double error = 1.0e-8;
	double *tl1 = GetCpuToLevel(0);
	delete[] tl1;
	//print_matrix(11, 11, tl1);
}
