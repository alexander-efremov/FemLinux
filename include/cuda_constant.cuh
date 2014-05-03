
#ifndef CUDA_CONSTS_H_INCLUDED
#define CUDA_CONSTS_H_INCLUDED
__constant__ double c_tau;
__constant__ double c_h;
__constant__ double c_b;
__constant__ double c_tau_to_current_time_level;
__constant__ double c_tau_to_h; // tau * h ( h = 1. / (p->x_size)) ;
__constant__ double c_lb;
__constant__ double c_rb;
__constant__ double c_ub;
__constant__ double c_bb;
__constant__ double c_tau_b;
__constant__ double c_pi_half;
__constant__ int c_x_length;
__constant__ int c_y_length;
__constant__ int c_n;
#endif