﻿#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "hemi.h"
#include "common.h"


double u_function_cuda(double par_b, double t, double x, double y)
{
    return  par_b * y * (1.-y) * (  C_pi_device /2. + atan( -x )  );
}

double v_function_cuda(
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double t, double x, double y )
{
    return	atan((x - lbDom) * (x - rbDom) * (1.+t) /10. * (y - ubDom) * (y - bbDom));
}

__device__ double d_u_function(double par_b, double t, double x,
										double y) {
											return par_b * y * (1. - y) * (C_pi_device / 2. + atan(-x));
}

 __device__ double d_v_function(double lbDom, double rbDom,
										double bbDom, double ubDom, double t, double x, double y) {
											return atan(
												(x - lbDom) * (x - rbDom) * (1. + t) / 10. * (y - ubDom)
												* (y - bbDom));
}

__device__ double d_itemOfInteg_1SpecType(
    double Py,
    double Qy,
    //
    double Gx,
    double Hx,
    //
    double a,
    double b )
{
    double integ;
    integ = (Hx - a)*(Hx - a)  -  (Gx - a)*(Gx - a);
    integ = integ * (  (Qy - b)*(Qy - b)  -  (Py - b)*(Py - b)  );
    return integ / 4.;
}

double h_analytSolut(double t, double x, double y )
{
    return 1.1  +  sin( t * x * y);
}

__device__ double d_analytSolut(double t, double x, double y )
{
    return 1.1  +  sin( t * x * y);
}

double d_initDataOfSol(ComputeParameters* p, int i, int j)
{
    return 
		1.1  +  sin( 0.* p->x[ i ] * p->y[ j ]);
}

double d_initDataOfSol(ComputeParameters p, int i, int j)
{
    return 
		1.1  +  sin( 0.* p.x[ i ] * p.y[ j ]);
}

double h_u_function_cuda(double par_b, double t, double x,
										double y) {
											return par_b * y * (1. - y) * (C_pi_device / 2. + atan(-x));
}

double h_v_function_cuda(double lbDom, double rbDom,
										double bbDom, double ubDom, double t, double x, double y) {
											return atan(
												(x - lbDom) * (x - rbDom) * (1. + t) / 10. * (y - ubDom)
												* (y - bbDom));
}


__device__ double d_itemOfInteg_2SpecType(
    double Py,
    double Qy,
    //
    double alpha,
    //
    double a,
    double b,
    double betta )
{
    double buf_D, integ;
    //   Computing...
    buf_D = (Qy - alpha) * (a*Qy + b - betta) * (a*Qy + b - betta) * (a*Qy + b - betta);
    buf_D = buf_D  -  (Py - alpha) * (a*Py + b - betta) * (a*Py + b - betta) * (a*Py + b - betta);
    integ = buf_D / (3. * a);
    buf_D = (a*Qy + b - betta) * (a*Qy + b - betta) * (a*Qy + b - betta) * (a*Qy + b - betta);
    buf_D = buf_D - (a*Py + b - betta) * (a*Py + b - betta) * (a*Py + b - betta) * (a*Py + b - betta);
    return integ  -  buf_D / (12. *a *a);
}

__device__ double d_integUnderLeftTr_OneCell(
    double par_a,                           //   -  Solution parameter.
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double Py,
    double Qy,
    //
    double a_SL,
    double b_SL,
    double Hx,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    int * indCurSqOx,                       //   -  Index of current square by Ox axis.
    int * indCurSqOy,                       //   -  Index of current square by Oy axis.
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV )
{
    double hx = masOX[1] - masOX[0];
    double hy = masOY[1] - masOY[0];
    double integ = 0;
    double buf_D, bufInteg_D;
    double rho[2][2];
    double t = tau * (iCurrTL - 1.);
    double x, y;
    if(  (indCurSqOx[0] >=0)  &&  (indCurSqOx[1] <=numOfOXSt)  ) {
        if(  (indCurSqOy[0] >=0)  &&  (indCurSqOy[1] <=numOfOYSt)  ) {
			//todo:  надо делить на numOfOXSt +1 потому что когда режу на части, то происходит выход за границы если делать так

            /*rho[0][0] = rhoInPrevTL_asV[ (numOfOXSt +1)*indCurSqOy[0] + indCurSqOx[0] ];
            rho[0][1] = rhoInPrevTL_asV[ (numOfOXSt +1)*indCurSqOy[1] + indCurSqOx[0] ];
            rho[1][0] = rhoInPrevTL_asV[ (numOfOXSt +1)*indCurSqOy[0] + indCurSqOx[1] ];
            rho[1][1] = rhoInPrevTL_asV[ (numOfOXSt +1)*indCurSqOy[1] + indCurSqOx[1] ];*/

			rho[0][0] = rhoInPrevTL_asV[ ((numOfOXSt +1)*indCurSqOy[0] + indCurSqOx[0]) % (numOfOXSt +1)];
            rho[0][1] = rhoInPrevTL_asV[ ((numOfOXSt +1)*indCurSqOy[1] + indCurSqOx[0])% (numOfOXSt +1) ];
            rho[1][0] = rhoInPrevTL_asV[ ((numOfOXSt +1)*indCurSqOy[0] + indCurSqOx[1])% (numOfOXSt +1) ];
            rho[1][1] = rhoInPrevTL_asV[ ((numOfOXSt +1)*indCurSqOy[1] + indCurSqOx[1])% (numOfOXSt +1) ];
        }
    }
    if(  (indCurSqOx[0] < 0)  ||  (indCurSqOx[1] > numOfOXSt)  ||  (indCurSqOy[0] < 0)  ||  (indCurSqOy[1] > numOfOYSt)  ) {
        x = indCurSqOx[0] * hx;
        y = indCurSqOy[0] * hy;
        rho[0][0]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[0] * hx;
        y = indCurSqOy[1] * hy;
        rho[0][1]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[1] * hx;
        y = indCurSqOy[0] * hy;
        rho[1][0]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[1] * hx;
        y = indCurSqOy[1] * hy;
        rho[1][1]  =  d_analytSolut(t, x, y );
    }

    //   1.
    buf_D = (Qy - masOY[ indCurSqOy[1] ]) * (Qy - masOY[ indCurSqOy[1] ])  -  (Py - masOY[ indCurSqOy[1] ]) * (Py - masOY[ indCurSqOy[1] ]);
    if(  (indCurSqOx[1] >= 0)  &&  (indCurSqOy[1] >= 0)  ) {
        buf_D = buf_D  *  (Hx - masOX[ indCurSqOx[1] ])  *  (Hx - masOX[ indCurSqOx[1] ]) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, masOY[ indCurSqOy[1] ], a_SL, b_SL, masOX[ indCurSqOx[1] ] );
    } else {
        buf_D = buf_D  *  (Hx -   hx * indCurSqOx[1]  )  *  (Hx -   hx * indCurSqOx[1]  ) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, hy * indCurSqOy[1], a_SL, b_SL,    hx * indCurSqOx[1]  );
    }
    buf_D = buf_D  -  bufInteg_D /2.;
    integ = buf_D * rho[0][0] /hx /hy;
    //   2.
    buf_D = (Qy - masOY[ indCurSqOy[1] ]) * (Qy - masOY[ indCurSqOy[1] ])  -  (Py - masOY[ indCurSqOy[1] ]) * (Py - masOY[ indCurSqOy[1] ]);
    if(  (indCurSqOx[0] >= 0)  &&  (indCurSqOy[1] >= 0)  ) {
        buf_D = -1. * buf_D  *  (Hx - masOX[ indCurSqOx[0] ])  *  (Hx - masOX[ indCurSqOx[0] ]) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, masOY[ indCurSqOy[1] ], a_SL, b_SL, masOX[ indCurSqOx[0] ] );
    } else {
        buf_D = -1. * buf_D  *  (Hx -   hx * indCurSqOx[0]  )  *  (Hx -   hx * indCurSqOx[0]  ) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, hy * indCurSqOy[1], a_SL, b_SL,   hx * indCurSqOx[0]   );
    }
    buf_D = buf_D  +  bufInteg_D /2.;
    integ = integ  +  buf_D * rho[1][0] /hx /hy;
    //   3.
    buf_D = (Qy - masOY[ indCurSqOy[0] ]) * (Qy - masOY[ indCurSqOy[0] ])  -  (Py - masOY[ indCurSqOy[0] ]) * (Py - masOY[ indCurSqOy[0] ]);
    if(  (indCurSqOx[1] >= 0)  &&  (indCurSqOy[0] >= 0)  ) {
        buf_D = -1. * buf_D  *  (Hx - masOX[ indCurSqOx[1] ])  *  (Hx - masOX[ indCurSqOx[1] ]) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, masOY[ indCurSqOy[0] ], a_SL, b_SL, masOX[ indCurSqOx[1] ] );
    } else {
        buf_D = -1. * buf_D  *  (Hx -   hx * indCurSqOx[1]  )  *  (Hx -   hx * indCurSqOx[1]  ) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, hy * indCurSqOy[0], a_SL, b_SL,   hx * indCurSqOx[1]   );
    }
    buf_D = buf_D  +  bufInteg_D /2.;
    integ = integ  +  buf_D * rho[0][1] /hx /hy;
    //   4.
    buf_D = (Qy - masOY[ indCurSqOy[0] ]) * (Qy - masOY[ indCurSqOy[0] ])  -  (Py - masOY[ indCurSqOy[0] ]) * (Py - masOY[ indCurSqOy[0] ]);
    if(  (indCurSqOx[0] >= 0)  &&  (indCurSqOy[0] >= 0)  ) {
        buf_D = buf_D  *  (Hx - masOX[ indCurSqOx[0] ])  *  (Hx - masOX[ indCurSqOx[0] ]) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, masOY[ indCurSqOy[0] ], a_SL, b_SL, masOX[ indCurSqOx[0] ] );
    } else {
        buf_D = buf_D  *  (Hx -   hx * indCurSqOx[0]  )  *  (Hx -   hx * indCurSqOx[0]  ) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, hy * indCurSqOy[0], a_SL, b_SL,   hx * indCurSqOx[0]   );
    }
    buf_D = buf_D  -  bufInteg_D /2.;
    return integ  +  buf_D * rho[1][1] /hx /hy;
}

__device__ double d_integUnderRightTr_OneCell(
    double par_a,                           //   -  Solution parameter.
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double Py,
    double Qy,
    //
    double a_SL,
    double b_SL,
    double Gx,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    int * indCurSqOx,                       //   -  Index of current square by Ox axis.
    int * indCurSqOy,                       //   -  Index of current square by Oy axis.
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV )
{
    return -1. * d_integUnderLeftTr_OneCell(
               par_a,                                  //   -  Solution parameter.
               //
               lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
               //
               bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
               //
               Py, Qy,
               //
               a_SL, b_SL,
               Gx,                                     //   -  double Hx,
               //
               tau, iCurrTL,                           //   -  Index of current time layer.
               //
               indCurSqOx,                             //   -  Index of current square by Ox axis.
               indCurSqOy,                             //   -  Index of current square by Oy axis.
               //
               masOX, numOfOXSt,                       //   -  Massive of OX steps. Dimension = numOfOXSt +1. Number of OX steps.
               //
               masOY, numOfOYSt,                       //   -  Massive of OY steps. Dimension = numOfOYSt +1. Number of OY steps.
               //
               rhoInPrevTL_asV );
}

__device__ double d_integUnderRectAng_OneCell(
    double par_a,                           //   -  Solution parameter.
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double Py,
    double Qy,
    //
    double Gx,
    double Hx,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    int * indCurSqOx,                       //   -  Index of current square by Ox axis.
    int * indCurSqOy,                       //   -  Index of current square by Oy axis.
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV )
{
    //   return ( fabs( (Qy - Py) * (Hx - Gx) ) );
    double hx = masOX[1] - masOX[0];
    double hy = masOY[1] - masOY[0];
    double integ = 0;
    double buf_D;
    double rho[2][2];
    double t = tau * (iCurrTL -1.);
    double x, y;
    if(   (indCurSqOx[0] >=0) && (indCurSqOy[0] >=0)  ) {
    /*    rho[0][0] = rhoInPrevTL_asV[ (numOfOXSt +1)*indCurSqOy[0] + indCurSqOx[0] ];
        rho[0][1] = rhoInPrevTL_asV[ (numOfOXSt +1)*indCurSqOy[1] + indCurSqOx[0] ];
        rho[1][0] = rhoInPrevTL_asV[ (numOfOXSt +1)*indCurSqOy[0] + indCurSqOx[1] ];
        rho[1][1] = rhoInPrevTL_asV[ (numOfOXSt +1)*indCurSqOy[1] + indCurSqOx[1] ];*/
		
			rho[0][0] = rhoInPrevTL_asV[ ((numOfOXSt +1)*indCurSqOy[0] + indCurSqOx[0]) % (numOfOXSt +1) ];
            rho[0][1] = rhoInPrevTL_asV[ ((numOfOXSt +1)*indCurSqOy[1] + indCurSqOx[0]) % (numOfOXSt +1) ];
            rho[1][0] = rhoInPrevTL_asV[ ((numOfOXSt +1)*indCurSqOy[0] + indCurSqOx[1]) % (numOfOXSt +1) ];
            rho[1][1] = rhoInPrevTL_asV[ ((numOfOXSt +1)*indCurSqOy[1] + indCurSqOx[1]) % (numOfOXSt +1) ];
    } else {
        x = indCurSqOx[0] * hx;
        y = indCurSqOy[0] * hy;
        rho[0][0]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[0] * hx;
        y = indCurSqOy[1] * hy;
        rho[0][1]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[1] * hx;
        y = indCurSqOy[0] * hy;
        rho[1][0]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[1] * hx;
        y = indCurSqOy[1] * hy;
        rho[1][1]  =  d_analytSolut(t, x, y );
    }

    if(   (indCurSqOx[1] >= 0) && (indCurSqOy[1] >= 0)   ) {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx, masOX[ indCurSqOx[1] ], masOY[ indCurSqOy[1] ] );
    } else {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx,   hx *indCurSqOx[1]   , hy * indCurSqOy[1] );
    }
    buf_D = buf_D  /hx /hy;
    integ = buf_D * rho[0][0];                            //   rhoInPrevTL[ indCurSqOx[0] ][ indCurSqOy[0] ];
    if(   (indCurSqOx[0] >= 0)  &&   (indCurSqOy[1] >= 0)   ) {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx, masOX[ indCurSqOx[0] ], masOY[ indCurSqOy[1] ] );
    } else {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx,   hx * indCurSqOx[0]  , hy * indCurSqOy[1] );
    }
    buf_D = buf_D  /hx /hy;
    integ = integ - buf_D * rho[1][0];                    //   rhoInPrevTL[ indCurSqOx[1] ][ indCurSqOy[0] ];
    if(   (indCurSqOx[1] >= 0)  &&  (indCurSqOy[0] >= 0)   ) {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx, masOX[ indCurSqOx[1] ], masOY[ indCurSqOy[0] ] );
    } else {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx,   hx * indCurSqOx[1]  , hy * indCurSqOy[0] );
    }
    buf_D = buf_D  /hx /hy;
    integ = integ - buf_D * rho[0][1];                    //   rhoInPrevTL[ indCurSqOx[0] ][ indCurSqOy[1] ];
    if(   (indCurSqOx[0] >= 0)  &&  (indCurSqOy[0] >= 0)   ) {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx, masOX[ indCurSqOx[0] ], masOY[ indCurSqOy[0] ] );
    } else {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx,   hx * indCurSqOx[0]  , hy * indCurSqOy[0] );
    }
    buf_D = buf_D  /hx /hy;
    return integ + buf_D * rho[1][1];                    //   rhoInPrevTL[ indCurSqOx[1] ][ indCurSqOy[1] ];
}


__device__ double d_integOfChan_SLRightSd(                         //   -  The domain is Channel with Slant Line on the right side.
    double par_a,                           //   -  Solution parameter.
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,   int wTrPCI,               //   -  Where travel point current (botton vertex) is.
    double *uv,   int wTrPNI,               //   -  Where travel point next (upper vertex) is.
    //
    int * indCurSqOx,                       //   -  Index by OX axis where bv and uv are.
    //
    double lb,  int * indLB,                //   -  Left boundary by Ox. Index by OX axis where lb is.
    //
    int * indCurSqOy,                       //   -  Index of current square by Oy axis.
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV )
{



    double mv[2], rv[2];                                  //   -  Middle and right vertices.
    int wMvI;                                             //   -  Where middle vertex is.
    int indCurSqOxToCh[2];                                //   -  Indices of current square by Ox axis to be changed. Under which we want to integrate.
    double h = masOX[1] - masOX[0];
    double a_SL, b_SL;                                    //   -  Coefficients of slant line: x = a_SL *y  +  b_SL.
    double Gx, Hx;                                        //   -  Left boundary for each integration.
    double integ = 0.;
    double buf_D;
    int j;

//   Let's compute helpful values.

    if( uv[0] <=  bv[0] ) {
        mv[0] = uv[0];
        mv[1] = uv[1];
        wMvI = wTrPNI;
        rv[0] = bv[0];
        rv[1] = bv[1];
    }

    if( uv[0] >  bv[0] ) {
        mv[0] = bv[0];
        mv[1] = bv[1];
        wMvI = wTrPCI;
        rv[0] = uv[0];
        rv[1] = uv[1];
    }



//   buf_D = fabs(mv[0] - lb) * fabs(uv[1] - bv[1])  +   fabs(uv[1] - bv[1]) * fabs(rv[0] - mv[0]) / 2.;
//   return  buf_D;


    if(  ( fabs(uv[1] - bv[1]) )  <=  1.e-12  ) {
        //   Computation is impossible. Too smale values. Let's return some approximate value.
        //   buf_D  =  (uv[1] - bv[1])  *  ((uv[0] + bv[0]) /2.  -  lb) * rhoInPrevTL[ indCurSqOx[0] ][ indCurSqOy[0] ];
        return fabs(uv[1] - bv[1]);   //   fabs(uv[1] - bv[1]);
    }


//   First step: from "lb" to "masOX[ indCurSqOx[0] ]" by iteration.
//   integ  += fabs( mv[0] - lb) * fabs(uv[1] - bv[1]);

    indCurSqOxToCh[0]  =  indLB[0];
    indCurSqOxToCh[1]  =  indCurSqOxToCh[0] +1;

    for( j = indLB[0]; j< indCurSqOx[0]; j++ ) {
        //   If this is first cell we should integrate under rectangle only.
        if( indCurSqOxToCh[0] >= 0 ) {
            Gx = masOX[ indCurSqOxToCh[0] ];
            Hx = masOX[ indCurSqOxToCh[1] ];
        }


        if( indCurSqOxToCh[0] < 0 ) {
            Gx = h * indCurSqOxToCh[0];
            Hx = h * indCurSqOxToCh[1];
        }

        if( j == indLB[0] ) {
            Gx = lb;
        }

        buf_D = d_integUnderRectAng_OneCell(
                    par_a,                                  //   -  Solution parameter.
                    //
                    lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                    //
                    bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                    //
                    bv[1],                                  //   -  double Py,
                    uv[1],                                  //   -  double Qy,
                    //
                    Gx,                                     //   -  double Gx,
                    Hx,                                     //   -  double Hx,
                    //
                    tau, iCurrTL,                           //   -  Index of current time layer.
                    //
                    indCurSqOxToCh,                         //   -  Index of current square by Ox axis.
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    //
                    masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                    numOfOXSt,                              //   -  Number of OX steps.
                    //
                    masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                    numOfOYSt,                              //   -  Number of OY steps.
                    //
                    rhoInPrevTL_asV );

        integ += buf_D;

        indCurSqOxToCh[0] +=  1;
        indCurSqOxToCh[1]  =  indCurSqOxToCh[0] +1;
    }

//   Integration. Second step: under [ indCurSqOx[0]; indCurSqOx[1] ] square.

//   A. Under rectangle.
    if(  wMvI == 1  ) {
        if( indCurSqOx[0] == indLB[0] ) {
            Gx = lb;
        }

        if( indCurSqOx[0] > indLB[0] ) {

            if( indCurSqOx[0] >= 0) {
                Gx = masOX[ indCurSqOx[0] ];
            }

            if( indCurSqOx[0] < 0) {
                Gx = h * indCurSqOx[0];
            }
        }

        buf_D = d_integUnderRectAng_OneCell(
                    par_a,                                  //   -  Solution parameter.
                    //
                    lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                    //
                    bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                    //
                    bv[1],                                  //   -  double Py,
                    uv[1],                                  //   -  double Qy,
                    //
                    Gx,                                     //   -  double Gx,
                    mv[0],                                  //   -  double Hx,
                    //
                    tau, iCurrTL,                           //   -  Index of current time layer.
                    //
                    indCurSqOx,                             //   -  Index of current square by Ox axis.
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    //
                    masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                    numOfOXSt,                              //   -  Number of OX steps.
                    //
                    masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                    numOfOYSt,                              //   -  Number of OY steps.
                    //
                    rhoInPrevTL_asV );

        integ += buf_D;
    }

//   B. Under triangle.

    if(  ( fabs(uv[1] - bv[1]) )  >  1.e-12  ) {
        //   integ += fabs(uv[1] - bv[1]) * (rv[0] - mv[0]) /2.;
        //   Coefficients of slant line: x = a_SL *y  +  b_SL.
        a_SL = (uv[0] - bv[0]) / (uv[1] - bv[1]);
        b_SL = bv[0] - a_SL * bv[1];


        //   Integration under one cell triangle.

        if( fabs( a_SL ) >  1.e-12 ) {
            buf_D = d_integUnderRightTr_OneCell(
                        par_a,                                  //   -  Solution parameter.
                        //
                        lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                        //
                        bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                        //
                        bv[1],                                  //   -  double Py,
                        uv[1],                                  //   -  double Qy,
                        //
                        a_SL,
                        b_SL,
                        mv[0],                                  //   -  double Gx,
                        //
                        tau, iCurrTL,                           //   -  Index of current time layer.
                        //
                        indCurSqOx,                             //   -  Index of current square by Ox axis.
                        indCurSqOy,                             //   -  Index of current square by Oy axis.
                        //
                        masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                        numOfOXSt,                              //   -  Number of OX steps.
                        //
                        masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                        numOfOYSt,                              //   -  Number of OY steps.
                        //
                        rhoInPrevTL_asV );

            integ += buf_D;
        }
    }
    return integ;
}


double h_bottomBound(ComputeParameters& p)
{
    return h_analytSolut(p.tau*p.currentTimeLevel, p.x[p.i], p.bb );
}

double h_bottomBound(ComputeParameters* p)
{
    return h_analytSolut(p->tau*p->currentTimeLevel, p->x[p->i], p->bb );
}

double h_upperBound(ComputeParameters& p)
{
    return  h_analytSolut(p.tau*p.currentTimeLevel, p.x[p.i],  p.ub );
}
double h_upperBound(ComputeParameters* p)
{
    return  h_analytSolut(p->tau*p->currentTimeLevel, p->x[p->i],  p->ub );
}

double h_rightBound(ComputeParameters& p)
{
    return h_analytSolut(p.tau*p.currentTimeLevel, p.rb, p.y[p.j] );
}

double h_rightBound(ComputeParameters* p)
{
    return h_analytSolut(p->tau*p->currentTimeLevel, p->rb, p->y[p->j] );
}

double h_leftBound(ComputeParameters& p)
{
    return h_analytSolut(p.tau*p.currentTimeLevel, p.lb, p.y[p.j]);
}

double h_leftBound(ComputeParameters* p)
{
    return h_analytSolut(p->tau*p->currentTimeLevel, p->lb, p->y[p->j]);
}

__device__ double d_integOfChan_SLLeftSd(                          //   -  The domain is Channel with Slant Line on the left side.
    double par_a,                           //   -  Solution parameter.
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,   int wTrPCI,               //   -  Where travel point current (botton vertex) is.
    double *uv,   int wTrPNI,               //   -  Where travel point next (upper vertex) is.
    //
    int * indCurSqOx,                       //   -  Index by OX axis where bv and uv are.
    //
    double rb,  int * indRB,                //   -  Right boundary by Ox. Index by OX axis where rb is.
    //
    int * indCurSqOy,                       //   -  Index of current square by Oy axis.
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV )
{



    double lv[2], mv[2];                                  //   -  Left and middle vertices.
    int wMvI;                                             //   -  Where middle vertex is.
    int indCurSqOxToCh[2];                                //   -  Indices of current square by Ox axis to be changed. Under which we want to integrate.
    double h = masOX[1] - masOX[0];
    double a_SL, b_SL;                                    //   -  Coefficients of slant line: x = a_SL *y  +  b_SL.
    double Gx, Hx;                                        //   -  Left and right boundary for each integration.
    double integ = 0.;
    double buf_D;
    int j;

//   Let's compute helpful values.

    if( uv[0] <=  bv[0] ) {
        lv[0] = uv[0];
        lv[1] = uv[1];
        mv[0] = bv[0];
        mv[1] = bv[1];
        wMvI = wTrPCI;
    }

    if( uv[0] >  bv[0] ) {
        lv[0] = bv[0];
        lv[1] = bv[1];
        mv[0] = uv[0];
        mv[1] = uv[1];
        wMvI = wTrPNI;
    }

    if(  ( fabs(uv[1] - bv[1]) )  <=  1.e-12  ) {
        //   Computation is impossible. Too smale values. Let's return some approximate value.
        //   buf_D  =  (uv[1] - bv[1])  *  (rb  - (uv[0] + bv[0]) /2.) * rhoInPrevTL[ indCurSqOx[0] ][ indCurSqOy[0] ];
        return fabs(uv[1] - bv[1]);   //   fabs(uv[1] - bv[1]);
    }

//   Integration. First step: under [ indCurSqOx[0]; indCurSqOx[1] ] square.

//   A. Under triangle.

    if(  ( fabs(uv[1] - bv[1]) )  >  1.e-12  ) {
        //   Coefficients of slant line: x = a_SL *y  +  b_SL.
        a_SL = (uv[0] - bv[0]) / (uv[1] - bv[1]);
        b_SL = bv[0] - a_SL * bv[1];

        //   Integration under one cell triangle.
        if( fabs( a_SL ) >  1.e-12 ) {
            buf_D = d_integUnderLeftTr_OneCell(
                        par_a,                                  //   -  Solution parameter.
                        //
                        lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                        //
                        bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                        //
                        bv[1],                                  //   -  double Py,
                        uv[1],                                  //   -  double Qy,
                        //
                        a_SL,
                        b_SL,
                        mv[0],                                  //   -  double Hx,
                        //
                        tau, iCurrTL,                           //   -  Index of current time layer.
                        //
                        indCurSqOx,                             //   -  Index of current square by Ox axis.
                        indCurSqOy,                             //   -  Index of current square by Oy axis.
                        //
                        masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                        numOfOXSt,                              //   -  Number of OX steps.
                        //
                        masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                        numOfOYSt,                              //   -  Number of OY steps.
                        //
                        rhoInPrevTL_asV );


            integ += buf_D;

        }
    }


//   B. Under rectangle. Need to be cheking.

    if(  wMvI == 1  ) {
        if( indCurSqOx[0] == indRB[0] ) {
            Hx = rb;
        }

        if( indCurSqOx[0] < indRB[0] ) {
            if( indCurSqOx[1] >= 0) {
                Hx = masOX[ indCurSqOx[1] ];
            }

            if( indCurSqOx[1] < 0) {
                Hx = h * indCurSqOx[1];
            }
        }

        buf_D = d_integUnderRectAng_OneCell(
                    par_a,                                  //   -  Solution parameter.
                    //
                    lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                    //
                    bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                    //
                    bv[1],                                  //   -  double Py,
                    uv[1],                                  //   -  double Qy,
                    //
                    mv[0],                                  //   -  double Gx,
                    Hx,                                     //   -  double Hx,
                    //
                    tau, iCurrTL,                           //   -  Index of current time layer.
                    //
                    indCurSqOx,                             //   -  Index of current square by Ox axis.
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    //
                    masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                    numOfOXSt,                              //   -  Number of OX steps.
                    //
                    masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                    numOfOYSt,                              //   -  Number of OY steps.
                    //
                    rhoInPrevTL_asV );

        integ += buf_D;
    }

//   Second step: from "masOX[ indCurSqOx[1] ]" to "rb" by iteration.


    indCurSqOxToCh[0]  =  indCurSqOx[0] +1;
    indCurSqOxToCh[1]  =  indCurSqOxToCh[0] +1;

    for( j = indCurSqOx[0] +1; j< indRB[0] +1; j++ ) {
        //   If this is first cell we should integrate under triangle only.

        if( indCurSqOxToCh[1] > 0 ) {
            Gx = masOX[ indCurSqOxToCh[0] ];
            Hx = masOX[ indCurSqOxToCh[1] ];
        }


        if( indCurSqOxToCh[1] <= 0 ) {
            Gx = h * indCurSqOxToCh[0];
            Hx = h * indCurSqOxToCh[1];
        }


        if( j == indRB[0] ) {
            Hx = rb;
        }


        buf_D = d_integUnderRectAng_OneCell(
                    par_a,                                  //   -  Solution parameter.
                    //
                    lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                    //
                    bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                    //
                    bv[1],                                  //   -  double Py,
                    uv[1],                                  //   -  double Qy,
                    //
                    Gx,                                     //   -  double Gx,
                    Hx,                                     //   -  double Hx,
                    //
                    tau, iCurrTL,                           //   -  Index of current time layer.
                    //
                    indCurSqOxToCh,                         //   -  Index of current square by Ox axis.
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    //
                    masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                    numOfOXSt,                              //   -  Number of OX steps.
                    //
                    masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                    numOfOYSt,                              //   -  Number of OY steps.
                    //
                    rhoInPrevTL_asV );

        integ += buf_D;



        indCurSqOxToCh[0] +=  1;
        indCurSqOxToCh[1]  =  indCurSqOxToCh[0] +1;
    }

    return integ;
}

__device__ double d_integUnderRigAngTr_BottLeft(
    double par_a,                           //   -  Solution parameter.
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,
    double *uv,
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV )
{
    double trPC[2];                                       //   -  Travel point current;
    int wTrPCI = 0;                                       //   -  Where travel point current is?
    double trPN[2];                                       //   -  Travel point next;
    int wTrPNI = 0;                                       //   -  Where travel point next is?
    double ang;                                           //   -  Angle of slant line. Should be greater zero.
    int indCurSqOx[2], indCurSqOy[2];                     //   -  Index of current square by Ox and Oy axes.
    int indRB[2];                                         //   -  Index of right boundary.
    double distOx, distOy;                                //   -  Distance to near Ox and Oy straight lines.
    bool isTrDone = false;                                //   -  Is travel done.
    double hx = masOX[1] - masOX[0];
    double hy = masOY[1] - masOY[0];
    double integOfBottTr = 0.;                            //   -  Value which we are computing.
    double buf_D;
    //   Initial data.
    trPC[0] = bv[0];
    trPC[1] = bv[1];
    if(  ( fabs(bv[0] - uv[0]) )  <  1.e-12  ) {
        //   This triangle has very small width. I guess further computation isn't correct.
        return fabs(bv[0] - uv[0]);
    }
    ang = (uv[1] - bv[1]) / (bv[0] - uv[0]);
    if(  fabs(ang)  <  1.e-12  ) {
        //   This triangle has very small height. I guess further computation isn't correct.
        return fabs(ang);
    }
    indCurSqOx[0] = (int)(  (trPC[0] - 1.e-14) /hx);      //   -  If trPC[0] is in grid edge I want it will be between in the left side of indCurSqOx[1].
    if( (trPC[0] - 1.e-14) <= 0 ) {
        indCurSqOx[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOx[1] = indCurSqOx[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    indRB[0] = indCurSqOx[0];
    indRB[1] = indRB[0] +1;
    indCurSqOy[0] = (int)(  (trPC[1] + 1.e-14) /hy);      //   -  If trPC[1] is in grid edge I want it will be between indCurSqOx[0] and indCurSqOx[1].
    if( (trPC[1] + 1.e-14) <= 0 ) {
        indCurSqOy[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOy[1] = indCurSqOy[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    if( indCurSqOx[0] >= 0) {
        distOx = trPC[0]  -  masOX[ indCurSqOx[0] ];
    }
    if( indCurSqOx[0] < 0 ) {
        distOx = fabs( trPC[0]  -  hx * indCurSqOx[0] );
    }
    if( indCurSqOy[1] >= 0 ) {
        distOy = masOY[ indCurSqOy[1] ]  -  trPC[1];
    }
    if( indCurSqOy[1] < 0 ) {
        distOy = fabs( hy * indCurSqOy[1]  -  trPC[1] );
    }
    do {
        //   a. First case.
        if( (distOy /distOx) <= ang ) {
            //   Across with straight line parallel Ox axis.
            wTrPNI = 1;
            if( indCurSqOy[1] >= 0) {
                trPN[1] = masOY[ indCurSqOy[1] ];
            }
            if( indCurSqOy[1] < 0) {
                trPN[1] = hy * indCurSqOy[1];
            }
            trPN[0] = bv[0] - (trPN[1] - bv[1]) /ang;
        }
        //   b. Second case.
        if( (distOy /distOx) > ang ) {
            //   Across with straight line parallel Oy axis.
            wTrPNI = 2;
            if( indCurSqOx[0] >= 0 ) {
                trPN[0]  =  masOX[ indCurSqOx[0] ];
            }
            if( indCurSqOx[0] < 0 ) {
                trPN[0]  =  hx * indCurSqOx[0];
            }
            trPN[1]  =  bv[1]  -  ang * (trPN[0] - bv[0]);
        }
        //   c. Cheking.
        if(  trPN[0]  <  (uv[0] + 1.e-14)  ) {
            trPN[0] = uv[0];
            trPN[1] = uv[1];
            isTrDone = true;
            wTrPNI = 0;
        }
        //   d. Integration.
        buf_D = d_integOfChan_SLLeftSd(                      //   -  The domain is Channel with Slant Line on the left side.
                    par_a,                                  //   -  Solution parameter.
                    //
                    lbDom, rbDom,
                    //
                    bbDom, ubDom,
                    //
                    tau, iCurrTL,                           //   -  Index of current time layer.
                    //
                    trPC,  wTrPCI,                          //   -  double *bv,
                    trPN,  wTrPNI,                          //   -  double *uv,
                    //
                    indCurSqOx,                             //   -  Indices where trPC and trPN are.
                    //
                    bv[0], indRB,                           //   -  double rb  =  Right boundary by Ox.
                    //
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    //
                    masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                    numOfOXSt,                              //   -  Number of OX steps.
                    //
                    masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                    numOfOYSt,                              //   -  Number of OY steps.
                    //
                    rhoInPrevTL_asV );
        integOfBottTr = integOfBottTr + buf_D;
        //   e. Updating.
        if( isTrDone == false ) {
            //   We will compute more. We need to redefine some values.
            wTrPCI = wTrPNI;
            trPC[0] = trPN[0];
            trPC[1] = trPN[1];
            if( wTrPNI == 1) {
                indCurSqOy[0] += 1;
                indCurSqOy[1] += 1;
            }
            if( wTrPNI == 2) {
                indCurSqOx[0] -= 1;
                indCurSqOx[1] -= 1;
            }
            if( indCurSqOx[0] >= 0) {
                distOx = trPC[0]  -  masOX[ indCurSqOx[0] ];
            }
            if( indCurSqOx[0] < 0) {
                distOx = fabs( trPC[0]  -  hx * indCurSqOx[0] );
            }
            if( indCurSqOy[1] >= 0 ) {
                distOy = masOY[ indCurSqOy[1] ]  -  trPC[1];
            }
            if( indCurSqOy[1] < 0 ) {
                distOy = fabs( hy * indCurSqOy[1]  -  trPC[1] );
            }
        }
    } while( !isTrDone );
    return integOfBottTr;
}

__device__ double d_integUnderRigAngTr_BottRight(
    double par_a,                           //   -  Solution parameter.
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,
    double *uv,
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV )
{
    double trPC[2];                                       //   -  Travel point current;
    int wTrPCI = 0;                                       //   -  Where travel point current is?
    double trPN[2];                                       //   -  Travel point next;
    int wTrPNI = 0;                                       //   -  Where travel point next is?
    double ang;                                           //   -  Angle of slant line. Should be greater zero.
    int indCurSqOx[2], indCurSqOy[2];                     //   -  Index of current square by Ox and Oy axes.
    int indLB[2];                                         //   -  Index of left boundary.
    double distOx, distOy;                                //   -  Distance to near Ox and Oy straight lines.
    bool isTrDone = false;                                //   -  Is travel done.
    double hx = masOX[1] - masOX[0];
    double hy = masOY[1] - masOY[0];
    double integOfBottTr = 0.;                            //   -  Value which we are computing.
    double buf_D;


    trPC[0] = bv[0];
    trPC[1] = bv[1];
    if(  ( fabs(bv[0] - uv[0]) )  <  1.e-12  ) return fabs(bv[0] - uv[0]);

    ang = (uv[1] - bv[1]) / (uv[0] - bv[0]);
    if(  fabs(ang)  <  1.e-12  ) return fabs(ang);

    indCurSqOx[0] = (int)(  (trPC[0] + 1.e-14) /hx);      //   -  If trPC[0] is in grid edge I want it will be between in the right side.

    if( (trPC[0] + 1.e-14) <= 0 )  indCurSqOx[0] -= 1;    //   -  The case when "trPC[0]" ia negative.

    indCurSqOx[1] = indCurSqOx[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    indLB[0] = indCurSqOx[0];
    indLB[1] = indLB[0] +1;
    indCurSqOy[0] = (int)(  (trPC[1] + 1.e-14) /hy);      //   -  If trPC[1] is in grid edge I want it will be in the upper side.
    if( (trPC[1] + 1.e-14) <= 0 ) {
        indCurSqOy[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOy[1] = indCurSqOy[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.

    if( indCurSqOx[1] >=0 ) {
        distOx = fabs( masOX[ indCurSqOx[1] ]  -  trPC[0] );
    }
    if( indCurSqOx[1] < 0 ) {
        distOx = fabs( hx * indCurSqOx[1]  -  trPC[0] );
    }
    if( indCurSqOy[1] >=0 ) {
        distOy = fabs( masOY[ indCurSqOy[1] ]  -  trPC[1] );
    }
    if( indCurSqOy[1] < 0 ) {
        distOy = fabs( hy * indCurSqOy[1]  -  trPC[1] );
    }
    do {
        //   a. First case.
        if( (distOy /distOx) <= ang ) {
            //   Across with straight line parallel Ox axis.
            wTrPNI = 1;
            if( indCurSqOy[1] >=0 ) {
                trPN[1] = masOY[ indCurSqOy[1] ];
            }
            if( indCurSqOy[1] < 0 ) {
                trPN[1] = hy * indCurSqOy[1];
            }
            trPN[0] = bv[0] + (trPN[1] - bv[1]) /ang;
        }
        //   b. Second case.
        if( (distOy /distOx) > ang ) {
            //   Across with straight line parallel Oy axis.
            wTrPNI = 2;
            if( indCurSqOx[1] >= 0 ) {
                trPN[0]  =  masOX[ indCurSqOx[1] ];
            }
            if( indCurSqOx[1]  < 0 ) {
                trPN[0]  =  hx * indCurSqOx[1];
            }
            trPN[1]  =  bv[1]  +  ang * (trPN[0] - bv[0]);
        }
        //   c. Cheking.
        if(  trPN[0]  >  (uv[0] - 1.e-14)  ) {             //   -  Without "fabs"!!!
            trPN[0] = uv[0];
            trPN[1] = uv[1];
            isTrDone = true;
            wTrPNI = 0;
        }
        //   d. Integration.
        buf_D = d_integOfChan_SLRightSd(                     //   -  The domain is Channel with Slant Line on the Right side.
                    par_a,                                  //   -  Solution parameter.
                    //
                    lbDom, rbDom,
                    //
                    bbDom, ubDom,
                    //
                    tau, iCurrTL,                           //   -  Index of current time layer.
                    //
                    trPC,  wTrPCI,                          //   -  double *bv,
                    trPN,  wTrPNI,                          //   -  double *uv,
                    //
                    indCurSqOx,                             //   -  Indices where trPC and trPN are.
                    //
                    bv[0], indLB,                           //   -  double lb  =  Left boundary by Ox.
                    //
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    //
                    masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                    numOfOXSt,                              //   -  Number of OX steps.
                    //
                    masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                    numOfOYSt,                              //   -  Number of OY steps.
                    //
                    rhoInPrevTL_asV );
        integOfBottTr = integOfBottTr + buf_D;
        //   e. Updating.
        if( isTrDone == false ) {
            //   We will compute more. We need to redefine some values.
            wTrPCI = wTrPNI;
            trPC[0] = trPN[0];
            trPC[1] = trPN[1];
            if( wTrPNI == 1) {
                indCurSqOy[0] += 1;
                indCurSqOy[1] += 1;
            }
            if( wTrPNI == 2) {
                indCurSqOx[0] += 1;
                indCurSqOx[1] += 1;
            }
            if( indCurSqOx[1] >=0 ) {
                distOx = fabs( masOX[ indCurSqOx[1] ]  -  trPC[0] );
            }
            if( indCurSqOx[1] < 0 ) {
                distOx = fabs( hx * indCurSqOx[1]  -  trPC[0] );
            }
            if( indCurSqOy[1] >=0 ) {
                distOy = fabs( masOY[ indCurSqOy[1] ]  -  trPC[1] );
            }
            if( indCurSqOy[1] < 0 ) {
                distOy = fabs( hy * indCurSqOy[1]  -  trPC[1] );
            }
        }
    } while( !isTrDone );
    return integOfBottTr;
}

__device__ double d_integUnderBottTr(
    double par_a,                           //   -  Item of left and right setback (parameter "a" in test).
    double par_b,                           //   -  Item of second parameter from "u_funcion".
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double * LvBt,                          //   -  Left, Right and Botton vertices of Botton triangle.
    double * RvBt,                          //   -  Left, Right and Botton vertices of Botton triangle.
    double * BvBt,                          //   -  Left, Right and Botton vertices of Botton triangle.
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV,
    int ii, int jj ) // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
{
    double integOfBottTr;
    double buf_D;
    //   Three ways are possible.
    //   1.
    if(  BvBt[0] <= LvBt[0]  ) {
        buf_D = d_integUnderRigAngTr_BottRight(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    BvBt, RvBt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfBottTr = buf_D;
        buf_D = d_integUnderRigAngTr_BottRight(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    BvBt, LvBt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfBottTr = integOfBottTr - buf_D;

//		printf("Bv<Lv: i= %d, j= %d      res= %le",ii,jj,integOfBottTr);  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return integOfBottTr;
    }
    //   2.
    if(  (BvBt[0] > LvBt[0]) && (BvBt[0] < RvBt[0]) ) {

        buf_D = d_integUnderRigAngTr_BottLeft(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    BvBt, LvBt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfBottTr = buf_D;

        buf_D = d_integUnderRigAngTr_BottRight(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    BvBt, RvBt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfBottTr = integOfBottTr + buf_D;

//		printf("Bv>Lv & Bv<Rv: i= %d, j= %d      res= %le",ii,jj,integOfBottTr);   // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return integOfBottTr;
    }
    //   3.
    if(  BvBt[0] >= RvBt[0]  ) {

        buf_D = d_integUnderRigAngTr_BottLeft(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    BvBt, LvBt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfBottTr = buf_D;
        buf_D = d_integUnderRigAngTr_BottLeft(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    BvBt, RvBt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfBottTr = integOfBottTr - buf_D;

//		printf("Bv>Rv: i= %d, j= %d      res= %le",ii,jj,integOfBottTr);     // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return integOfBottTr;
    }
    return integOfBottTr;
}

__device__ double d_integUnderRigAngTr_UppLeft(
    double par_a,                           //   -  Solution parameter.
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,
    double *uv,
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV )
{
    //   return ( fabs( (uv[1] - bv[1]) * (bv[0] - uv[0]) /2.) );
    double trPC[2];                                       //   -  Travel point current;
    int wTrPCI = 0;                                       //   -  Where travel point current is?
    double trPN[2];                                       //   -  Travel point next;
    int wTrPNI = 0;                                       //   -  Where travel point next is?
    double ang;                                           //   -  Angle of slant line. Should be greater zero.
    int indCurSqOx[2], indCurSqOy[2];                     //   -  Index of current square by Ox and Oy axes.
    int indRB[2];                                         //   -  Index of right boundary.
    double distOx, distOy;                                //   -  Distance to near Ox and Oy straight lines.
    bool isTrDone = false;                                //   -  Is travel done.
    double hx = masOX[1] - masOX[0];
    double hy = masOY[1] - masOY[0];
    double integOfUppTr = 0.;                             //   -  Value which we are computing.
    double buf_D;
    //   Initial data.
    trPC[0] = bv[0];
    trPC[1] = bv[1];
    if(  ( fabs(bv[0] - uv[0]) )  <  1.e-12  )	return fabs(bv[0] - uv[0]);

    ang = (uv[1] - bv[1]) / (uv[0] - bv[0]);
    if(  fabs(ang)  <  1.e-12  ) return fabs(ang);

    //   The follow equations are quite important.
    indCurSqOx[0] = (int)(  (trPC[0] + 1.e-14) /hx);      //   -  If trPC[0] is in grid edge I want it will be in the right side.
    if( (trPC[0] + 1.e-14) <= 0 ) {
        indCurSqOx[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOx[1] = indCurSqOx[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    indCurSqOy[0] = (int)(  (trPC[1] + 1.e-14) /hy);      //   -  If trPC[1] is in grid edge I want it will be in the upper square.
    if( (trPC[1] + 1.e-14) <= 0 ) {
        indCurSqOy[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOy[1] = indCurSqOy[0] +1;
    indRB[0] = (int)(  (uv[0] - 1.e-14) /hy);             //   -  If uv[0] is in grid edge I want it will be in the left side.
    if( (uv[0] - 1.e-14) <= 0 ) {
        indRB[0] -= 1;     //   -  The case when "trPC[0]" ia negative.
    }
    indRB[1] = indRB[0] +1;
    if( indCurSqOx[1] >= 0) {
        distOx = masOX[ indCurSqOx[1] ]  -  trPC[0];
    }
    if( indCurSqOx[1] < 0) {
        distOx = fabs( hx * indCurSqOx[1]  -  trPC[0] );
    }
    if( indCurSqOy[1] >= 0 ) {
        distOy = masOY[ indCurSqOy[1] ]  -  trPC[1];
    }
    if( indCurSqOy[1] < 0 ) {
        distOy = fabs( hy * indCurSqOy[1]  -  trPC[1] );
    }
    do {
        //   a. First case.
        if( (distOy /distOx) <= ang ) {
            //   Across with straight line parallel Ox axis.
            wTrPNI = 1;
            if( indCurSqOy[1] >= 0 ) {
                trPN[1] = masOY[ indCurSqOy[1] ];
            }
            if( indCurSqOy[1] < 0 ) {
                trPN[1] = hy * indCurSqOy[1];
            }
            trPN[0] = bv[0] + (trPN[1] - bv[1]) /ang;
        }
        //   b. Second case.
        if( (distOy /distOx) > ang ) {
            //   Across with straight line parallel Oy axis.
            wTrPNI = 2;
            if( indCurSqOx[1] >= 0 ) {
                trPN[0]  =  masOX[ indCurSqOx[1] ];
            }
            if( indCurSqOx[1] < 0 ) {
                trPN[0]  =  hx * indCurSqOx[1];
            }
            trPN[1]  =  bv[1]  +  ang * (trPN[0] - bv[0]);
        }
        //   c. Cheking.
        if(  trPN[0]  >  (uv[0] - 1.e-14)  ) {
            trPN[0] = uv[0];
            trPN[1] = uv[1];
            isTrDone = true;
            wTrPNI = 0;
        }
        //   d. Integration.
        buf_D = d_integOfChan_SLLeftSd(                      //   -  The domain is Channel with Slant Line on the left side.
                    par_a,                                  //   -  Solution parameter.
                    //
                    lbDom, rbDom,
                    //
                    bbDom, ubDom,
                    //
                    tau, iCurrTL,                           //   -  Index of current time layer.
                    //
                    trPC,  wTrPCI,                          //   -  double *bv,
                    trPN,  wTrPNI,                          //   -  double *uv,
                    //
                    indCurSqOx,                             //   -  Indices where trPC and trPN are.
                    //
                    uv[0], indRB,                           //   -  double rb  =  Right boundary by Ox.
                    //
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    //
                    masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                    numOfOXSt,                              //   -  Number of OX steps.
                    //
                    masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                    numOfOYSt,                              //   -  Number of OY steps.
                    //
                    rhoInPrevTL_asV );
        integOfUppTr = integOfUppTr + buf_D;
        //   e. Updating.
        if( isTrDone == false ) {
            //   We will compute more. We need to redefine some values.
            wTrPCI = wTrPNI;
            trPC[0] = trPN[0];
            trPC[1] = trPN[1];
            if( wTrPNI == 1) {
                indCurSqOy[0] += 1;
                indCurSqOy[1] += 1;
            }
            if( wTrPNI == 2) {
                indCurSqOx[0] += 1;
                indCurSqOx[1] += 1;
            }
            if( indCurSqOx[1] >= 0) {
                distOx = fabs( masOX[ indCurSqOx[1] ]  -  trPC[0] );
            }
            if( indCurSqOx[1] < 0) {
                distOx = fabs( hx * indCurSqOx[1]  -  trPC[0] );
            }
            if( indCurSqOy[1] >= 0 ) {
                distOy = fabs( masOY[ indCurSqOy[1] ]  -  trPC[1] );
            }
            if( indCurSqOy[1] < 0 ) {
                distOy = fabs( hy * indCurSqOy[1]  -  trPC[1] );
            }
        }
    } while( !isTrDone );
    return integOfUppTr;
}

__device__ double d_integUnderRigAngTr_UppRight(
    double par_a,                           //   -  Solution parameter.
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,
    double *uv,
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV )
{
    //   return ( fabs( (uv[1] - bv[1]) * (bv[0] - uv[0]) /2.) );
    double trPC[2];                                       //   -  Travel point current;
    int wTrPCI = 0;                                       //   -  Where travel point current is?
    double trPN[2];                                       //   -  Travel point next;
    int wTrPNI = 0;                                       //   -  Where travel point next is?
    double ang;                                           //   -  Angle of slant line. Should be greater zero.
    int indCurSqOx[2], indCurSqOy[2];                     //   -  Index of current square by Ox and Oy axes.
    int indLB[2];                                         //   -  Index of left boundary.
    double distOx, distOy;                                //   -  Distance to near Ox and Oy straight lines.
    bool isTrDone = false;                                //   -  Is travel done.
    double hx = masOX[1] - masOX[0];
    double hy = masOY[1] - masOY[0];
    double integOfUppTr = 0.;                             //   -  Value which we are computing.
    double buf_D;
    //   Initial data.
    trPC[0] = bv[0];
    trPC[1] = bv[1];
    if(  ( fabs(bv[0] - uv[0]) )  <  1.e-12  ) {
        //   This triangle has very small width. I guess further computation isn't correct.
        return fabs(bv[0] - uv[0]);
    }
    ang = (uv[1] - bv[1]) / (bv[0] - uv[0]);
    if(  fabs(ang)  <  1.e-12  ) {
        //   This triangle has very small height. I guess further computation isn't correct.
        return fabs(ang);
    }
    indCurSqOx[0] = (int)(  (trPC[0] - 1.e-14) /hx);      //   -  If trPC[0] is in grid edge I want it will be between in the left side.
    if( (trPC[0] - 1.e-14) <= 0 ) {
        indCurSqOx[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOx[1] = indCurSqOx[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    indLB[0] = (int)( (uv[0] + 1.e-14) /hx);
    if( (uv[0] + 1.e-14) <=0 ) {
        indLB[0] -= 1;     //   -  The case when "trPC[0]" ia negative.
    }
    indLB[1] = indLB[0] +1;
    indCurSqOy[0] = (int)(  (trPC[1] + 1.e-14) /hy);      //   -  If trPC[1] is in grid edge I want it will be in the upper side.
    if( (trPC[1] + 1.e-14) <= 0 ) {
        indCurSqOy[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOy[1] = indCurSqOy[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    if( indCurSqOx[0] >= 0 ) {
        distOx = fabs( trPC[0]  -  masOX[ indCurSqOx[0] ] );
    }
    if( indCurSqOx[0] < 0 ) {
        distOx = fabs( trPC[0]  -  hx * indCurSqOx[0] );
    }
    if( indCurSqOy[1] >= 0 ) {
        distOy = fabs( masOY[ indCurSqOy[1] ]  -  trPC[1] );
    }
    if( indCurSqOy[1] < 0 ) {
        distOy = fabs( hy * indCurSqOy[1]  -  trPC[1] );
    }
    do {
        //   a. First case.
        if( (distOy /distOx) <= ang ) {
            //   Across with straight line parallel Ox axis.
            wTrPNI = 1;
            if( indCurSqOy[1] >= 0 ) {
                trPN[1] = masOY[ indCurSqOy[1] ];
            }
            if( indCurSqOy[1] < 0 ) {
                trPN[1] = hy * indCurSqOy[1];
            }
            trPN[0] = bv[0] - (trPN[1] - bv[1]) /ang;
        }
        //   b. Second case.
        if( (distOy /distOx) > ang ) {
            //   Across with straight line parallel Oy axis.
            wTrPNI = 2;
            if( indCurSqOx[0] >= 0 ) {
                trPN[0]  =  masOX[ indCurSqOx[0] ];
            }
            if( indCurSqOx[0] < 0 ) {
                trPN[0]  =  hx * indCurSqOx[0];
            }
            trPN[1]  =  bv[1]  -  ang * (trPN[0] - bv[0]);
        }
        //   c. Cheking.
        if(  trPN[0]  <  (uv[0] + 1.e-14)  ) {
            trPN[0] = uv[0];
            trPN[1] = uv[1];
            isTrDone = true;
            wTrPNI = 0;
        }
        //   d. Integration.
        buf_D = d_integOfChan_SLRightSd(                     //   -  The domain is Channel with Slant Line on the Right side.
                    par_a,                                  //   -  Solution parameter.
                    //
                    lbDom, rbDom,
                    //
                    bbDom, ubDom,
                    //
                    tau, iCurrTL,                           //   -  Index of current time layer.
                    //
                    trPC,  wTrPCI,                          //   -  double *bv,
                    trPN,  wTrPNI,                          //   -  double *uv,
                    //
                    indCurSqOx,                             //   -  Indices where trPC and trPN are.
                    //
                    uv[0], indLB,                           //   -  double lb  =  Left boundary by Ox.
                    //
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    //
                    masOX,                                  //   -  Massive of OX steps. Dimension = numOfOXSt +1.
                    numOfOXSt,                              //   -  Number of OX steps.
                    //
                    masOY,                                  //   -  Massive of OY steps. Dimension = numOfOYSt +1.
                    numOfOYSt,                              //   -  Number of OY steps.
                    //
                    rhoInPrevTL_asV );
        integOfUppTr = integOfUppTr + buf_D;
        //   e. Updating.
        if( isTrDone == false ) {
            //   We will compute more. We need to redefine some values.
            wTrPCI = wTrPNI;
            trPC[0] = trPN[0];
            trPC[1] = trPN[1];
            if( wTrPNI == 1) {
                indCurSqOy[0] += 1;
                indCurSqOy[1] += 1;
            }
            if( wTrPNI == 2) {
                indCurSqOx[0] -= 1;
                indCurSqOx[1] -= 1;
            }
            if( indCurSqOx[0] >= 0 ) {
                distOx = fabs( trPC[0]  -  masOX[ indCurSqOx[0] ] );
            }
            if( indCurSqOx[0] < 0 ) {
                distOx = fabs( trPC[0]  -  hx * indCurSqOx[0] );
            }
            if( indCurSqOy[1] >= 0 ) {
                distOy = fabs( masOY[ indCurSqOy[1] ]  -  trPC[1] );
            }
            if( indCurSqOy[1] < 0 ) {
                distOy = fabs( hy * indCurSqOy[1]  -  trPC[1] );
            }
        }
    } while(!isTrDone);
    return integOfUppTr;
}

__device__ double d_integUnderUpperTr(
    double par_a,                           //   -  Item of left and right setback (parameter "a" in test).
    double par_b,                           //   -  Item of second parameter from "u_funcion".
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double * LvUt,                          //   -  Left, Right and Upper vertices of Upper triangle.
    double * RvUt,                          //   -  Left, Right and Upper vertices of Upper triangle.
    double * UvUt,                          //   -  Left, Right and Upper vertices of Upper triangle.
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY step
    //
    double * rhoInPrevTL_asV )
{
    double integOfUppTr;
    double buf_D;
    //   Three ways are possible.
    //   1.
    if(  UvUt[0] <= LvUt[0]  ) {
        buf_D = d_integUnderRigAngTr_UppRight(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    RvUt, UvUt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfUppTr = buf_D;
        buf_D = d_integUnderRigAngTr_UppRight(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    LvUt, UvUt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfUppTr = integOfUppTr - buf_D;
        return integOfUppTr;
    }
    //   2.
    if(  (UvUt[0] > LvUt[0]) && (UvUt[0] < RvUt[0]) ) {
        buf_D = d_integUnderRigAngTr_UppLeft(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    LvUt, UvUt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfUppTr = buf_D;

        buf_D = d_integUnderRigAngTr_UppRight(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    RvUt, UvUt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfUppTr = integOfUppTr + buf_D;
        return integOfUppTr;
    }
    //   3.
    if(  UvUt[0] >= RvUt[0]  ) {
        buf_D = d_integUnderRigAngTr_UppLeft(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    LvUt, UvUt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfUppTr = buf_D;
        buf_D = d_integUnderRigAngTr_UppLeft(
                    par_a,   lbDom, rbDom,   bbDom, ubDom,   tau, iCurrTL,
                    //
                    RvUt, UvUt,   masOX, numOfOXSt, masOY, numOfOYSt,   rhoInPrevTL_asV );
        integOfUppTr = integOfUppTr - buf_D;
        return integOfUppTr;
    }
    return integOfUppTr;
}

__device__ double d_integUnderUnunifTr(
    double par_a,                           //   -  Item of left and right setback (parameter "a" in test).
    double par_b,                           //   -  Item of second parameter from "u_funcion".
    //
    double lbDom,                           //   -  Left and right boundaries of rectangular domain.
    double rbDom,
    //
    double bbDom,                           //   -  Botton and upper boundaries of rectangular domain.
    double ubDom,
    //
    double tau,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double * firVer,                        //   -  First vertex of triangle.
    double * secVer,                        //   -  Second vertex of triangle.
    double * thiVer,                        //   -  Third vertex of triangle.
    //
    const double * masOX,                         //   -  Massive of OX steps. Dimension = numOfOXSt +1.
    int numOfOXSt,                          //   -  Number of OX steps.
    //
    const double * masOY,                         //   -  Massive of OY steps. Dimension = numOfOYSt +1.
    int numOfOYSt,                          //   -  Number of OY steps.
    //
    double * rhoInPrevTL_asV,
    int ii, int jj ) //!!!!!!!!!!!!!!!!!!!
{
    double bv[2], mv[2], uv[2];                           //   -  Botton, middle and upper vertices of triangle.
    bool isFirVUsed = false;
    bool isSecVUsed = false;
    bool isThiVUsed = false;
    bool is1VUsed, is2VUsed, is3VUsed;
    double a_LC, b_LC, c_LC;                              //   -  Coefficients of line betweeen "bv" and "uv" vertices.
    double ap[2];                                         //   -  Across point of line through "bv" to "uv" and "y == mv[1]"
    double LvBt[2], RvBt[2], BvBt[2];                     //   -  Left, Right and Botton vertices of Botton triangle.
    double integOfBottTr;                                 //   -  Item of integral under Botton triangle.
    double LvUt[2], RvUt[2], UvUt[2];                     //   -  Left, Right and Upper vertices of Upper triangle.
    double integOfUppTr;                                  //   -  Item of integral under Upper triangle.
    double integ = 0.;                                    //   -  Item which I'm computing.
    //   1. I need to understand which vertex is botton, middle and upper.
    bv[1] = firVer[1];
    bv[0] = firVer[0];
    isFirVUsed = true;
    if( bv[1] > secVer[1] ) {
        bv[1] = secVer[1];
        bv[0] = secVer[0];
        isFirVUsed = false;
        isSecVUsed = true;
    }
    if( bv[1] > thiVer[1] ) {
        bv[1] = thiVer[1];
        bv[0] = thiVer[0];
        isFirVUsed = false;
        isSecVUsed = false;
        isThiVUsed = true;
    }
    uv[1] = masOY[0];                                     //   -  The minimum possible value.
    is1VUsed = false;
    is2VUsed = false;
    is3VUsed = false;
    if(  (uv[1] < firVer[1])  &&  (isFirVUsed == false)  ) {
        uv[1] = firVer[1];
        uv[0] = firVer[0];
        is1VUsed = true;
    }
    if(  (uv[1] < secVer[1])  &&  (isSecVUsed == false)  ) {
        uv[1] = secVer[1];
        uv[0] = secVer[0];
        is2VUsed = true;
        is1VUsed = false;
    }
    if(  (uv[1] < thiVer[1])  &&  (isThiVUsed == false)  ) {
        uv[1] = thiVer[1];
        uv[0] = thiVer[0];
        is3VUsed = true;
        is2VUsed = false;
        is1VUsed = false;
    }
    //   Dangerous.
    if(  (isFirVUsed == false) &&  (is1VUsed == false)  ) {
        mv[1] = firVer[1];
        mv[0] = firVer[0];
    }
    if(  (isSecVUsed == false) &&  (is2VUsed == false)  ) {
        mv[1] = secVer[1];
        mv[0] = secVer[0];
    }
    if(  (isThiVUsed == false) &&  (is3VUsed == false)  ) {
        mv[1] = thiVer[1];
        mv[0] = thiVer[0];
    }
    //   2. I want to compute across point.
    //   2.a Let's compute line coefficients betweeen "bv" and "uv" vertices.
    //   a_LC * x  +  b_LC * y  = c_LC.
    a_LC  =  uv[1] - bv[1];
    b_LC  =  bv[0] - uv[0];
    c_LC  =  (bv[0] - uv[0])*bv[1]  +  (uv[1] - bv[1])*bv[0];
    //   2.b Across point.
    ap[1]  =  mv[1];
    if( fabs(a_LC) < 1.e-12 ) {
        //   This triangle has very small height. I guess further computation isn't correct.
        return 1.e-12;
    }
    ap[0]  =  (c_LC  -  b_LC * ap[1])  /a_LC;

//	printf("i= %d, j= %d : ap[0]= %le      mv[0]= %le \n",ii,jj, ap[0], mv[0]); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    //   3. There the middle vertex relativly straight line is? Two ways are possible.
    if( mv[0] < ap[0] ) {
        //   Left, Right and Botton vertices of Botton triangle.
        LvBt[0]  =  mv[0];
        LvBt[1]  =  mv[1];
        RvBt[0]  =  ap[0];
        RvBt[1]  =  ap[1];
        BvBt[0]  =  bv[0];
        BvBt[1]  =  bv[1];
        integOfBottTr = d_integUnderBottTr(
                            par_a, par_b,
                            //
                            lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                            //
                            bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                            //
                            tau, iCurrTL,                           //   -  Index of current time layer.
                            //
                            LvBt, RvBt, BvBt,                       //   -  Left, Right and Botton vertices of Botton triangle.
                            //
                            masOX, numOfOXSt,                       //   -  Number of OX steps.
                            //
                            masOY, numOfOYSt,                       //   -  Number of OY steps.
                            //
                            rhoInPrevTL_asV,
                            ii, jj ); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integ = integOfBottTr;

//		printf("m<a:   i= %d, j= %d : integ= %le \n",ii,jj, integ); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        //   Left, Right and Upper vertices of Upper triangle.
        LvUt[0]  =  mv[0];
        LvUt[1]  =  mv[1];
        RvUt[0]  =  ap[0];
        RvUt[1]  =  ap[1];
        UvUt[0]  =  uv[0];
        UvUt[1]  =  uv[1];
        integOfUppTr = d_integUnderUpperTr(
                           par_a, par_b,
                           //
                           lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                           //
                           bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                           //
                           tau, iCurrTL,                           //   -  Index of current time layer.
                           //
                           LvUt, RvUt, UvUt,                       //   -  Left, Right and Botton vertices of Upper triangle.
                           //
                           masOX, numOfOXSt,                       //   -  Number of OX steps.
                           //
                           masOY, numOfOYSt,                       //   -  Number of OY steps.
                           //
                           rhoInPrevTL_asV );
        integ = integ + integOfUppTr;
        return integ;
    }
    if( mv[0] >= ap[0] ) {
        //   Left, Right and Botton vertices of Botton triangle.
        LvBt[0]  =  ap[0];
        LvBt[1]  =  ap[1];
        RvBt[0]  =  mv[0];
        RvBt[1]  =  mv[1];
        BvBt[0]  =  bv[0];
        BvBt[1]  =  bv[1];
        integOfBottTr = d_integUnderBottTr(
                            par_a, par_b,
                            //
                            lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                            //
                            bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                            //
                            tau, iCurrTL,                           //   -  Index of current time layer.
                            //
                            LvBt, RvBt, BvBt,                       //   -  Left, Right and Botton vertices of Botton triangle.
                            //
                            masOX, numOfOXSt,                       //   -  Number of OX steps.
                            //
                            masOY, numOfOYSt,                       //   -  Number of OY steps.
                            //
                            rhoInPrevTL_asV,
                            ii, jj ); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integ = integOfBottTr;

//		printf("m>a:   i= %d, j= %d : integ= %le \n",ii,jj, integ);  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        //   Left, Right and Upper vertices of Upper triangle.
        LvUt[0]  =  ap[0];
        LvUt[1]  =  ap[1];
        RvUt[0]  =  mv[0];
        RvUt[1]  =  mv[1];
        UvUt[0]  =  uv[0];
        UvUt[1]  =  uv[1];
        integOfUppTr = d_integUnderUpperTr(
                           par_a, par_b,
                           //
                           lbDom, rbDom,                           //   -  Left and right boundaries of rectangular domain.
                           //
                           bbDom, ubDom,                           //   -  Botton and upper boundaries of rectangular domain.
                           //
                           tau, iCurrTL,                           //   -  Index of current time layer.
                           //
                           LvUt, RvUt, UvUt,                       //   -  Left, Right and Botton vertices of Upper triangle.
                           //
                           masOX, numOfOXSt,                       //   -  Number of OX steps.
                           //
                           masOY, numOfOYSt,                       //   -  Number of OY steps.
                           //
                           rhoInPrevTL_asV );
        return integ + integOfUppTr;
    }
    return integ;
}





__device__ double d_f_function(ComputeParameters p, const int currentTimeLevel, const int i, const int j)
{
    double t  =  p.tau * currentTimeLevel;
    double x  =  p.x[ i ];
    double y  =  p.y[ j ];

    double arg_v  =  (x - p.lb) * (x - p.rb) * (1.+t) /10. * (y - p.ub) * (y - p.bb);
    double rho, dRhoDT, dRhoDX, dRhoDY;
    double u, duDX;
    double v, dvDY;
    rho  =  d_analytSolut(t, x, y );
    dRhoDT  =  x * y * cos( t*x*y );
    dRhoDX  =  t * y * cos( t*x*y );
    dRhoDY  =  t * x * cos( t*x*y );
    u  =  d_u_function(p.b, t, x, y );
    duDX  = -p.b * y * (1.-y)  /  ( 1.  +  x * x );
    v  =  d_v_function(p.lb, p.rb, p.bb, p.ub, t, x, y );
    dvDY  =  (x - p.lb) * (x - p.rb) * (1.+t) /10. * (y - p.bb + y - p.ub);
    dvDY  =  dvDY  /  ( 1.  +  arg_v * arg_v );
    return dRhoDT   +   rho * duDX   +   u * dRhoDX   +   rho * dvDY   +   v * dRhoDY;
}


double h_f_function(ComputeParameters p, const int currentTimeLevel, const int i, const int j)
{
    double t  =  p.tau * currentTimeLevel;
    double x  =  p.x[ i ];
    double y  =  p.y[ j ];

    double arg_v  =  (x - p.lb) * (x - p.rb) * (1.+t) /10. * (y - p.ub) * (y - p.bb);
    double rho, dRhoDT, dRhoDX, dRhoDY;
    double u, duDX;
    double v, dvDY;
    rho  =  h_analytSolut(t, x, y );
    dRhoDT  =  x * y * cos( t*x*y );
    dRhoDX  =  t * y * cos( t*x*y );
    dRhoDY  =  t * x * cos( t*x*y );
    u  =  h_u_function_cuda(p.b, t, x, y );
    duDX  = -p.b * y * (1.-y)  /  ( 1.  +  x * x );
    v  =  h_v_function_cuda(p.lb, p.rb, p.bb, p.ub, t, x, y );
    dvDY  =  (x - p.lb) * (x - p.rb) * (1.+t) /10. * (y - p.bb + y - p.ub);
    dvDY  =  dvDY  /  ( 1.  +  arg_v * arg_v );
    return dRhoDT   +   rho * duDX   +   u * dRhoDX   +   rho * dvDY   +   v * dRhoDY;
}

double h_f_function(ComputeParameters* p, const int currentTimeLevel, const int i, const int j)
{
    double t  =  p->tau * currentTimeLevel;
    double x  =  p->x[ i ];
    double y  =  p->y[ j ];

    double arg_v  =  (x - p->lb) * (x - p->rb) * (1.+t) /10. * (y - p->ub) * (y - p->bb);
    double rho, dRhoDT, dRhoDX, dRhoDY;
    double u, duDX;
    double v, dvDY;
    rho  =  h_analytSolut(t, x, y );
    dRhoDT  =  x * y * cos( t*x*y );
    dRhoDX  =  t * y * cos( t*x*y );
    dRhoDY  =  t * x * cos( t*x*y );
    u  =  h_u_function_cuda(p->b, t, x, y );
    duDX  = -p->b * y * (1.-y)  /  ( 1.  +  x * x );
    v  =  h_v_function_cuda(p->lb, p->rb, p->bb, p->ub, t, x, y );
    dvDY  =  (x - p->lb) * (x - p->rb) * (1.+t) /10. * (y - p->bb + y - p->ub);
    dvDY  =  dvDY  /  ( 1.  +  arg_v * arg_v );
    return dRhoDT   +   rho * duDX   +   u * dRhoDX   +   rho * dvDY   +   v * dRhoDY;
}

void h_quadrAngleType(ComputeParameters* p, double* first_x1, double* second_x1, double* third_x1, double* first_x2, double*
	second_x2, double* third_x2,
	double* first_y1, double* second_y1, double* third_y1, double* first_y2, double* second_y2, double* third_y2)
{
   

    double alpha[2], betta[2], gamma[2], theta[2];        //   -  Vertexes of square. Anticlockwise order from left botton vertex.
    double u, v;                                          //   -  Items of velocity components.
    double alNew[2], beNew[2], gaNew[2], thNew[2];        //   -  New positions of vertexes. Vertexes of quadrangle.
   

    //   1. First of all let's compute coordinates of square vertexes.

    //  OX:

    if( p->i == 0 ) {
        alpha[0]  =  p->x[ p->i ];
        betta[0]  =  ( p->x[p->i]  +  p->x[p->i +1] ) /2.;
        gamma[0]  =  ( p->x[p->i]  +  p->x[p->i +1] ) /2.;
        theta[0]  =  p->x[ p->i ];
    }

    if( p->i == p->x_size ) {
        alpha[0]  =  ( p->x[p->i -1]  +  p->x[p->i] ) /2.;
        betta[0]  =  p->x[ p->i ];
        gamma[0]  =  p->x[ p->i ];
        theta[0]  =  ( p->x[p->i -1]  +  p->x[p->i] ) /2.;
    }

    if( (p->i > 0)  &&  (p->i < p->x_size) ) {
        alpha[0]  =  ( p->x[p->i -1]  +  p->x[p->i] ) /2.;
        betta[0]  =  ( p->x[p->i +1]  +  p->x[p->i] ) /2.;
        gamma[0]  =  ( p->x[p->i +1]  +  p->x[p->i] ) /2.;
        theta[0]  =  ( p->x[p->i -1]  +  p->x[p->i] ) /2.;
    }

    //  OY:
    if( p->j == 0 ) {
        alpha[1]  =  p->y[ p->j ];
        betta[1]  =  p->y[ p->j ];
        gamma[1]  =  ( p->y[p->j]  +  p->y[ p->j +1] ) /2.;
        theta[1]  =  ( p->y[p->j]  +  p->y[ p->j +1] ) /2.;
    }

    if( p->j == p->y_size ) {
        alpha[1]  =  ( p->y[p->j]  +  p->y[ p->j -1] ) /2.;
        betta[1]  =  ( p->y[p->j]  +  p->y[ p->j -1] ) /2.;
        gamma[1]  =  p->y[ p->j ];
        theta[1]  =  p->y[ p->j ];
    }

    if( (p->j > 0) && (p->j < p->y_size) ) {
        alpha[1]  =  ( p->y[p->j]  +  p->y[ p->j -1] ) /2.;
        betta[1]  =  ( p->y[p->j]  +  p->y[ p->j -1] ) /2.;
        gamma[1]  =  ( p->y[p->j]  +  p->y[ p->j +1] ) /2.;
        theta[1]  =  ( p->y[p->j]  +  p->y[ p->j +1] ) /2.;
    }

    //   2. Now let's compute new coordinates on the previous time level of alpha, betta, gamma, theta points.
    //  alNew.

    u = u_function_cuda( p->b,   p->currentTimeLevel * p->tau, alpha[0], alpha[1] );
    v = v_function_cuda( p->lb, p->rb,   p->bb, p->ub,   p->currentTimeLevel * p->tau, alpha[0], alpha[1] );
    alNew[0]  =  alpha[0]  -  p->tau * u;
    alNew[1]  =  alpha[1]  -  p->tau * v;

    //  beNew.
    u = u_function_cuda( p->b,   p->currentTimeLevel * p->tau, betta[0], betta[1] );
    v = v_function_cuda( p->lb, p->rb, p->bb, p->ub, p->currentTimeLevel * p->tau, betta[0], betta[1] );
    beNew[0]  =  betta[0]  -  p->tau * u;
    beNew[1]  =  betta[1]  -  p->tau * v;

    //  gaNew.

    u = u_function_cuda( p->b, p->currentTimeLevel * p->tau, gamma[0], gamma[1] );
    v = v_function_cuda( p->lb, p->rb, p->bb, p->ub, p->currentTimeLevel * p->tau, gamma[0], gamma[1] );
    gaNew[0]  =  gamma[0]  -  p->tau * u;
    gaNew[1]  =  gamma[1]  -  p->tau * v;

    //  thNew.
    u = u_function_cuda( p->b,   p->currentTimeLevel * p->tau, theta[0], theta[1] );
    v = v_function_cuda( p->lb, p->rb, p->bb, p->ub,   p->currentTimeLevel * p->tau, theta[0], theta[1] );
    thNew[0]  =  theta[0]  -  p->tau * u;
    thNew[1]  =  theta[1]  -  p->tau * v;

	*first_x1 = alNew[0];
	*first_y1 = alNew[1];
	*second_x1 = beNew[0];
	*second_y1 = beNew[1];
	*third_x1 = gaNew[0];
	*third_y1 = gaNew[1];
	
	*first_x2 = alNew[0];
	*first_y2 = alNew[1];
	*second_x2 = thNew[0];
	*second_y2 = thNew[1];
	*third_x2 = gaNew[0];
	*third_y2 = gaNew[1];
    
}

double d_normOfMatrAtL1_asV(
    const double *masOX,                          //   -  Massive of OX grid nodes. Dimension = dimOX.
    int dimOX,
    //
    const double *masOY,                          //   -  Massive of OY grid nodes. Dimension = dimOY.
    int dimOY,
    //
    double * mat_asV )
{
    double norm = 0.;
    double hx  =  masOX[1]  -  masOX[0];
    double hy  =  masOY[1]  -  masOY[0];

    for( int k=1; k< dimOY -1; k++ ) {
        for( int j=1; j< dimOX -1; j++ ) {
            norm  +=  fabs( mat_asV[ dimOX*k + j ] );
        }
    }
    return hx * hy * norm;
}

__global__ void kernel(ComputeParameters p, double *rhoInPrevTL_asV, double* result, int length, int x_length, int part_number, int chunk)
{
//	const int offset = hemiGetElementOffset();
//	const int stride = hemiGetElementStride();
//
//	for (int opt = offset; opt < length; opt += stride) {
//		p.i = (opt % x_length + 1) ;
//		p.j = (opt / x_length + 1) + (int) (part_number*chunk / x_length);
//#ifdef PRINT_VALUES
//		printf("p.i = %d p.j = %d \n", p.i, p.j);
//#endif
//		 double buf_D = d_integUnderUnunifTr(
//                       p.a, p.b,
//                       //
//                       p.lb, p.rb,
//                       //
//                       p.bb, p.ub,
//                       //
//                       p.tau, p.currentTimeLevel,
//                       //
//					   first[opt].first, first[opt].second, first[opt].third,
//                       //
//                       p.x, p.x_size,
//                       //
//                       p.y, p.y_size,
//                       rhoInPrevTL_asV,
//                       p.i, p.j);
//
//
//     buf_D += d_integUnderUnunifTr(
//               p.a, p.b,                           //   -  Analitycal solution parameters.
//               //
//               p.lb, p.rb,                           //   -  Left and right boundaries of rectangular domain.
//               //
//               p.bb, p.ub,                           //   -  Botton and upper boundaries of rectangular domain.
//               //
//               p.tau, p.currentTimeLevel,                           //   -  Index of current time layer.
//               //
//			   second[opt].first, second[opt].second, second[opt].third,           //   -  Vertices of first triangle.
//               //
//               p.x, p.x_size,                       //   -  Number of OX steps.
//               //
//               p.y, p.y_size,                       //   -  Number of OY steps.
//               //
//               rhoInPrevTL_asV,
//               p.i, p.j );
//	 result[opt] = buf_D;
	//}
}

//#define PRINT_VALUES



double d_spaceVolumeInPrevTL(ComputeParameters p, double *rhoInPrevTL_asV )
{
	
   /* Triangle firstT = Triangle();
    Triangle secondT = Triangle();*/
	return 0;
  //  if( h_quadrAngleType(p, firstT, secondT ) != 1 )	{
		//// TODO: write exception
  //      return -1.;
  //  }

    //double buf_D = d_integUnderUnunifTr(
    //                   p.a, p.b,
    //                   //
    //                   p.lb, p.rb,
    //                   //
    //                   p.bb, p.ub,
    //                   //
    //                   p.tau, p.currentTimeLevel,
    //                   //
    //                   firstT.first, firstT.second, firstT.third,
    //                   //
    //                   p.x, p.x_size,
    //                   //
    //                   p.y, p.y_size,
    //                   rhoInPrevTL_asV,
    //                   p.i, p.j);


    //return buf_D + d_integUnderUnunifTr(
               //p.a, p.b,                           //   -  Analitycal solution parameters.
               ////
               //p.lb, p.rb,                           //   -  Left and right boundaries of rectangular domain.
               ////
               //p.bb, p.ub,                           //   -  Botton and upper boundaries of rectangular domain.
               ////
               //p.tau, p.currentTimeLevel,                           //   -  Index of current time layer.
               ////
               //secondT.first, secondT.second, secondT.third,           //   -  Vertices of first triangle.
               ////
               //p.x, p.x_size,                       //   -  Number of OX steps.
               ////
               //p.y, p.y_size,                       //   -  Number of OY steps.
               ////
               //rhoInPrevTL_asV,
               //p.i, p.j );
}


double d_solByEqualVolumes(ComputeParameters p)
{
    double *rhoInPrevTL_asV;
    double spVolInPrevTL;

    rhoInPrevTL_asV = new double [ p.size ];
    //   Initial data of rho.
    for( int k = 0; k <= p.x_size; k++ ) {
        for( int j = 0; j <= p.y_size; j++ ) {
            rhoInPrevTL_asV[ (p.x_size+1)*k + j ]  =  d_initDataOfSol(p, j, k);
        }
    }

    for( int currentTimeLevel = 1; currentTimeLevel <= p.t_count; currentTimeLevel++ ) {

        std::cout<<"\r \t \t \t \t \t \t \t \t \r";
        std::cout << " Nx = " <<  p.x_size;
        std::cout << ", indexOfCurrTL = " << currentTimeLevel << " ( " << p.t_count << " ) " << std::flush;

       
        //  Compute boundaries
        for(int i = 0; i  <= p.x_size; ++i) {
            p.i = i;
            p.result[ i  ]  =   h_bottomBound(p);
            p.result[ (p.x_size + 1)*p.y_size + i  ]  =   h_upperBound(p);
        }
        for( int j = 0; j <= p.y_size; ++j) {
            p.j = j;
            p.result[ (p.x_size + 1)*j ]  =  h_leftBound(p);
            p.result[ (p.x_size + 1)*j  +  p.x_size ]  =  h_rightBound(p);
        }

		
        for( int j = 1; j < p.y_size; j++ ) {
            for( int i = 1; i < p.x_size; i++ ) {

                p.currentTimeLevel = currentTimeLevel;
                p.i = i;
                p.j = j;

                spVolInPrevTL  =  d_spaceVolumeInPrevTL(p, rhoInPrevTL_asV );

                double buf_D  =  (p.x[i + 1]  -  p.x[i - 1]) /2.;
                spVolInPrevTL  =  spVolInPrevTL / buf_D;

                buf_D  =  (p.y[j + 1]  -  p.y[j - 1]) /2.;
                spVolInPrevTL  =  spVolInPrevTL / buf_D;

                p.result[ (p.x_size + 1)*j + i ]  =  spVolInPrevTL;
                p.result[ (p.x_size + 1)*j + i ] +=  p.tau * h_f_function(p, currentTimeLevel, i, j);
            }
        }

        for( int i = 0; i < p.size; i++ ) {
            rhoInPrevTL_asV[ i ]  =  p.result[ i ];
        }
    }

    delete[] rhoInPrevTL_asV;
    std::cout << std::endl;
    return 1;
}

float solve_cuda_params(ComputeParameters p)
{
double *rhoInPrevTL_asV;
    double spVolInPrevTL;

    rhoInPrevTL_asV = new double [ p.size ];
    //   Initial data of rho.
    for( int k = 0; k <= p.x_size; k++ ) {
        for( int j = 0; j <= p.y_size; j++ ) {
            rhoInPrevTL_asV[ (p.x_size+1)*k + j ]  =  d_initDataOfSol(p, j, k);
        }
    }

    for( int currentTimeLevel = 1; currentTimeLevel <= p.t_count; currentTimeLevel++ ) {

        std::cout<<"\r \t \t \t \t \t \t \t \t \r";
        std::cout << " Nx = " <<  p.x_size;
        std::cout << ", indexOfCurrTL = " << currentTimeLevel << " ( " << p.t_count << " ) " << std::flush;

       
        //  Compute boundaries
        for(int i = 0; i  <= p.x_size; ++i) {
            p.i = i;
            p.result[ i  ]  =   h_bottomBound(p);
            p.result[ (p.x_size + 1)*p.y_size + i  ]  =   h_upperBound(p);
        }
        for( int j = 0; j <= p.y_size; ++j) {
            p.j = j;
            p.result[ (p.x_size + 1)*j ]  =  h_leftBound(p);
            p.result[ (p.x_size + 1)*j  +  p.x_size ]  =  h_rightBound(p);
        }

		
        for( int j = 1; j < p.y_size; j++ ) {
            for( int i = 1; i < p.x_size; i++ ) {

                p.currentTimeLevel = currentTimeLevel;
                p.i = i;
                p.j = j;

                spVolInPrevTL  =  d_spaceVolumeInPrevTL(p, rhoInPrevTL_asV );

                double buf_D  =  (p.x[i + 1]  -  p.x[i - 1]) /2.;
                spVolInPrevTL  =  spVolInPrevTL / buf_D;

                buf_D  =  (p.y[j + 1]  -  p.y[j - 1]) /2.;
                spVolInPrevTL  =  spVolInPrevTL / buf_D;

                p.result[ (p.x_size + 1)*j + i ]  =  spVolInPrevTL;
                p.result[ (p.x_size + 1)*j + i ] +=  p.tau * h_f_function(p, currentTimeLevel, i, j);
            }
        }

        for( int i = 0; i < p.size; i++ ) {
            rhoInPrevTL_asV[ i ]  =  p.result[ i ];
        }
    }

    delete[] rhoInPrevTL_asV;
    std::cout << std::endl;
    return 1;
    return 1.0;
}

