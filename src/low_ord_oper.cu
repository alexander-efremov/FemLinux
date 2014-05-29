#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "math.h"
#include "hemi.h"
#include "common.h"
#include "cuda_constant.cuh"


__device__ double d_u_function(double t, double x,
                                        double y) {
                                            return c_b * y * (1. - y) * (C_pi_device / 2. + atan(-x));
}

__device__ double d_v_function(double t, double x, double y) {
                                            return atan(
                                                (x - c_lb) * (x - c_rb) * (1. + t) / 10. * (y - c_ub)
                                                * (y - c_bb));
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

__device__ double d_analytSolut(double t, double x, double y )
{
    return 1.1  +  sin( t * x * y);
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
    double Py,
    double Qy,
    //
    double a_SL,
    double b_SL,
    double Hx,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    int * indCurSqOx,                       //   -  Index of current square by Ox axis.
    int * indCurSqOy,                       //   -  Index of current square by Oy axis.
    double * rhoInPrevTL_asV )
{
    double integ = 0;
    double buf_D, bufInteg_D;
    double rho[2][2];
    double t = c_tau * (iCurrTL - 1.);
    double x, y;
    if(  (indCurSqOx[0] >=0)  &&  (indCurSqOx[1] <= c_x_length)  ) {
        if(  (indCurSqOy[0] >=0)  &&  (indCurSqOy[1] <=c_y_length)  ) {

            rho[0][0] = rhoInPrevTL_asV[ ((c_x_length +1)*indCurSqOy[0] + indCurSqOx[0]) ];
            rho[0][1] = rhoInPrevTL_asV[ ((c_x_length +1)*indCurSqOy[1] + indCurSqOx[0]) ];
            rho[1][0] = rhoInPrevTL_asV[ ((c_x_length +1)*indCurSqOy[0] + indCurSqOx[1]) ];
            rho[1][1] = rhoInPrevTL_asV[ ((c_x_length +1)*indCurSqOy[1] + indCurSqOx[1]) ];
        }
    }
    if(  (indCurSqOx[0] < 0)  ||  (indCurSqOx[1] > c_x_length)  ||  (indCurSqOy[0] < 0)  ||  (indCurSqOy[1] > c_y_length)  ) {
        x = indCurSqOx[0] * c_h;
        y = indCurSqOy[0] * c_h;
        rho[0][0]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[0] * c_h;
        y = indCurSqOy[1] * c_h;
        rho[0][1]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[1] * c_h;
        y = indCurSqOy[0] * c_h;
        rho[1][0]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[1] * c_h;
        y = indCurSqOy[1] * c_h;
        
        rho[1][1]  =  d_analytSolut(t, x, y );
    }

    //   1.
    buf_D = (Qy - c_h * indCurSqOy[1]) * (Qy - c_h * indCurSqOy[1])  -  (Py - c_h * indCurSqOy[1]) * (Py - c_h * indCurSqOy[1]);
    if(  (indCurSqOx[1] >= 0)  &&  (indCurSqOy[1] >= 0)  ) {
        buf_D = buf_D  *  (Hx - c_h * indCurSqOx[1])  *  (Hx -  c_h *  indCurSqOx[1]) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, c_h * indCurSqOy[1], a_SL, b_SL, c_h *  indCurSqOx[1]  );
    } else {
        buf_D = buf_D  *  (Hx -   c_h * indCurSqOx[1]  )  *  (Hx -   c_h * indCurSqOx[1]  ) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, c_h * indCurSqOy[1], a_SL, b_SL,    c_h * indCurSqOx[1]  );
    }
    buf_D = buf_D  -  bufInteg_D /2.;
    integ = buf_D * rho[0][0] /c_h /c_h;


    //   2.
    buf_D = (Qy - c_h * indCurSqOy[1]) * (Qy - c_h * indCurSqOy[1])  -  (Py - c_h * indCurSqOy[1]) * (Py - c_h *  indCurSqOy[1]);
    if(  (indCurSqOx[0] >= 0)  &&  (indCurSqOy[1] >= 0)  ) {
        buf_D = -1. * buf_D  *  (Hx - c_h *  indCurSqOx[0])  *  (Hx - c_h *  indCurSqOx[0]) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, c_h * indCurSqOy[1], a_SL, b_SL, c_h * indCurSqOx[0] );
    } else {
        buf_D = -1. * buf_D  *  (Hx -   c_h * indCurSqOx[0]  )  *  (Hx -   c_h * indCurSqOx[0]  ) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, c_h * indCurSqOy[1], a_SL, b_SL,  c_h * indCurSqOx[0]   );
    }
    buf_D = buf_D  +  bufInteg_D /2.;
    integ = integ  +  buf_D * rho[1][0] /c_h /c_h;

    
    //   3.
    buf_D = (Qy - c_h *  indCurSqOy[0]) * (Qy - c_h * indCurSqOy[0])  -  (Py - c_h * indCurSqOy[0]) * (Py - c_h * indCurSqOy[0]);
    if(  (indCurSqOx[1] >= 0)  &&  (indCurSqOy[0] >= 0)  ) {
        buf_D = -1. * buf_D  *  (Hx - c_h *  indCurSqOx[1])  *  (Hx - c_h * indCurSqOx[1]) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, c_h * indCurSqOy[0], a_SL, b_SL, c_h * indCurSqOx[1] );
    } else {
        buf_D = -1. * buf_D  *  (Hx -   c_h * indCurSqOx[1]  )  *  (Hx -   c_h * indCurSqOx[1]  ) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, c_h * indCurSqOy[0], a_SL, b_SL,   c_h * indCurSqOx[1]   );
    }
    buf_D = buf_D  +  bufInteg_D /2.;
    integ = integ  +  buf_D * rho[0][1] /c_h /c_h;
    //   4.
    buf_D = (Qy - c_h *  indCurSqOy[0]) * (Qy - c_h *  indCurSqOy[0])  -  (Py - c_h *  indCurSqOy[0]) * (Py - c_h * indCurSqOy[0]);
    if(  (indCurSqOx[0] >= 0)  &&  (indCurSqOy[0] >= 0)  ) {
        buf_D = buf_D  *  (Hx - c_h *  indCurSqOx[0])  *  (Hx - c_h * indCurSqOx[0]) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, c_h * indCurSqOy[0], a_SL, b_SL, c_h * indCurSqOx[0] );
    } else {
        buf_D = buf_D  *  (Hx -   c_h * indCurSqOx[0]  )  *  (Hx -   c_h * indCurSqOx[0]  ) /4.;
        bufInteg_D = d_itemOfInteg_2SpecType( Py, Qy, c_h * indCurSqOy[0], a_SL, b_SL,   c_h * indCurSqOx[0]   );
    }
    buf_D = buf_D  -  bufInteg_D /2.;
    integ +=  buf_D * rho[1][1] /c_h /c_h;
    
    return integ;
}

__device__ double d_integUnderRightTr_OneCell( 
    double Py,
    double Qy,
    //
    double a_SL,
    double b_SL,
    double Gx,
    int iCurrTL,                            //   -  Index of current time layer.
    //
    int * indCurSqOx,                       //   -  Index of current square by Ox axis.
    int * indCurSqOy,                       //   -  Index of current square by Oy axis.
     
    double * rhoInPrevTL_asV )
{
    return -1. * d_integUnderLeftTr_OneCell( 
               Py, Qy,
               //
               a_SL, b_SL,
               Gx,                                     //   -  double Hx,
               iCurrTL,                           //   -  Index of current time layer.
               //
               indCurSqOx,                             //   -  Index of current square by Ox axis.
               indCurSqOy,                             //   -  Index of current square by Oy axis.
               //
                
               rhoInPrevTL_asV );
}

__device__ double d_integUnderRectAng_OneCell( 
    double Py,
    double Qy,
    //
    double Gx,
    double Hx,

    int iCurrTL,                            //   -  Index of current time layer.
    //
    int * indCurSqOx,                       //   -  Index of current square by Ox axis.
    int * indCurSqOy,                       //   -  Index of current square by Oy axis.
    
  
    double * rhoInPrevTL_asV )
{   
    double integ = 0;
    double buf_D;
    double rho[2][2];
    double t = c_tau * (iCurrTL -1.);
    double x, y;
    if(   (indCurSqOx[0] >=0) && (indCurSqOy[0] >=0)  ) {
        
            rho[0][0] = rhoInPrevTL_asV[ ((c_x_length +1)*indCurSqOy[0] + indCurSqOx[0])  ];
            rho[0][1] = rhoInPrevTL_asV[ ((c_x_length +1)*indCurSqOy[1] + indCurSqOx[0]) ];
            rho[1][0] = rhoInPrevTL_asV[ ((c_x_length +1)*indCurSqOy[0] + indCurSqOx[1])  ];
            rho[1][1] = rhoInPrevTL_asV[ ((c_x_length +1)*indCurSqOy[1] + indCurSqOx[1])  ];
    } else {
        x = indCurSqOx[0] * c_h;
        y = indCurSqOy[0] * c_h;
        rho[0][0]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[0] * c_h;
        y = indCurSqOy[1] * c_h;
        rho[0][1]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[1] * c_h;
        y = indCurSqOy[0] * c_h;
        rho[1][0]  =  d_analytSolut(t, x, y );
        x = indCurSqOx[1] * c_h;
        y = indCurSqOy[1] * c_h;
        rho[1][1]  =  d_analytSolut(t, x, y );
    }

    if(   (indCurSqOx[1] >= 0) && (indCurSqOy[1] >= 0)   ) {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx, c_h * indCurSqOx[1],  c_h * indCurSqOy[1] );
    } else {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx,   c_h *indCurSqOx[1]   , c_h * indCurSqOy[1] );
    }
    buf_D = buf_D  /c_h /c_h;
    integ = buf_D * rho[0][0];                            //   rhoInPrevTL[ indCurSqOx[0] ][ indCurSqOy[0] ];
    if(   (indCurSqOx[0] >= 0)  &&   (indCurSqOy[1] >= 0)   ) {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx, c_h *indCurSqOx[0] , c_h * indCurSqOy[1]  );
    } else {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx,   c_h * indCurSqOx[0]  , c_h * indCurSqOy[1] );
    }
    buf_D = buf_D  /c_h /c_h;
    integ = integ - buf_D * rho[1][0];                    //   rhoInPrevTL[ indCurSqOx[1] ][ indCurSqOy[0] ];
    if(   (indCurSqOx[1] >= 0)  &&  (indCurSqOy[0] >= 0)   ) {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx, c_h * indCurSqOx[1] , c_h * indCurSqOy[0]  );
    } else {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx,   c_h * indCurSqOx[1]  , c_h * indCurSqOy[0] );
    }
    buf_D = buf_D  /c_h /c_h;
    integ = integ - buf_D * rho[0][1];                    //   rhoInPrevTL[ indCurSqOx[0] ][ indCurSqOy[1] ];
    if(   (indCurSqOx[0] >= 0)  &&  (indCurSqOy[0] >= 0)   ) {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx, c_h *indCurSqOx[0], c_h * indCurSqOy[0] );
    } else {
        buf_D = d_itemOfInteg_1SpecType( Py, Qy, Gx, Hx,   c_h * indCurSqOx[0]  , c_h * indCurSqOy[0] );
    }
    buf_D = buf_D  /c_h /c_h;
   

    return integ + buf_D * rho[1][1];                    //   rhoInPrevTL[ indCurSqOx[1] ][ indCurSqOy[1] ];
}

__device__ double d_integOfChan_SLRightSd( 
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
    
    double * rhoInPrevTL_asV )
{
    double mv[2];                                  //   -  Middle and right vertices.
    int wMvI;                                             //   -  Where middle vertex is.
    int indCurSqOxToCh[2];                                //   -  Indices of current square by Ox axis to be changed. Under which we want to integrate.
    double h = c_h;
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
        
    }

    if( uv[0] >  bv[0] ) {
        mv[0] = bv[0];
        mv[1] = bv[1];
        wMvI = wTrPCI;
         
    }

    if(  ( fabs(uv[1] - bv[1]) )  <=  1.e-12  ) {
        //   Computation is impossible. Too smale values. Let's return some approximate value.
        //   buf_D  =  (uv[1] - bv[1])  *  ((uv[0] + bv[0]) /2.  -  lb) * rhoInPrevTL[ indCurSqOx[0] ][ indCurSqOy[0] ];
        return fabs(uv[1] - bv[1]);   //   fabs(uv[1] - bv[1]);
    }


//   First step: from "lb" to "mas OX[ indCurSqOx[0] ]" by iteration.
//   integ  += fabs( mv[0] - lb) * fabs(uv[1] - bv[1]);

    indCurSqOxToCh[0]  =  indLB[0];
    indCurSqOxToCh[1]  =  indCurSqOxToCh[0] +1;

    for( j = indLB[0]; j< indCurSqOx[0]; j++ ) {
        //   If this is first cell we should integrate under rectangle only.
        if( indCurSqOxToCh[0] >= 0 ) {
            Gx = c_h *  indCurSqOxToCh[0];
            Hx = c_h *  indCurSqOxToCh[1];
        }


        if( indCurSqOxToCh[0] < 0 ) {
            Gx = h * indCurSqOxToCh[0];
            Hx = h * indCurSqOxToCh[1];
        }

        if( j == indLB[0] ) {
            Gx = lb;
        }

        buf_D = d_integUnderRectAng_OneCell( 
                    bv[1],                                  //   -  double Py,
                    uv[1],                                  //   -  double Qy,
                    //
                    Gx,                                     //   -  double Gx,
                    Hx,                                     //   -  double Hx,
                    //
                      iCurrTL,                           //   -  Index of current time layer.
                    //
                    indCurSqOxToCh,                         //   -  Index of current square by Ox axis.
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                 
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
                Gx = c_h *  indCurSqOx[0];
            }

            if( indCurSqOx[0] < 0) {
                Gx = h * indCurSqOx[0];
            }
        }

        buf_D = d_integUnderRectAng_OneCell( 
                    bv[1],                                  //   -  double Py,
                    uv[1],                                  //   -  double Qy,
                    //
                    Gx,                                     //   -  double Gx,
                    mv[0],                                  //   -  double Hx,
                    //
                   iCurrTL,                           //   -  Index of current time layer.
                    //
                    indCurSqOx,                             //   -  Index of current square by Ox axis.
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    
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
                        bv[1],                                  //   -  double Py,
                        uv[1],                                  //   -  double Qy,
                        //
                        a_SL,
                        b_SL,
                        mv[0],                                  //   -  double Gx,
                        //
                         iCurrTL,                           //   -  Index of current time layer.
                        //
                        indCurSqOx,                             //   -  Index of current square by Ox axis.
                        indCurSqOy,                             //   -  Index of current square by Oy axis.
                         
                        
                        rhoInPrevTL_asV );
            
            integ += buf_D;
        }
    }
    return integ;
}

__device__ double d_integOfChan_SLLeftSd(   
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
    
    double * rhoInPrevTL_asV )
{
    double   mv[2];                                  //   -  Left and middle vertices.
    int wMvI;                                             //   -  Where middle vertex is.
    int indCurSqOxToCh[2];                                //   -  Indices of current square by Ox axis to be changed. Under which we want to integrate.
    double h = c_h;
    double a_SL, b_SL;                                    //   -  Coefficients of slant line: x = a_SL *y  +  b_SL.
    double Gx, Hx;                                        //   -  Left and right boundary for each integration.
    double integ = 0.;
    double buf_D;
    int j;

//   Let's compute helpful values.

    if( uv[0] <=  bv[0] ) {
         
        mv[0] = bv[0];
        mv[1] = bv[1];
        wMvI = wTrPCI;
    }

    if( uv[0] >  bv[0] ) {
        
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
                        bv[1],                                  //   -  double Py,
                        uv[1],                                  //   -  double Qy,
                        //
                        a_SL,
                        b_SL,
                        mv[0],                                  //   -  double Hx,
                        //
                         iCurrTL,                           //   -  Index of current time layer.
                        //
                        indCurSqOx,                             //   -  Index of current square by Ox axis.
                        indCurSqOy,                             //   -  Index of current square by Oy axis.
                         
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
                Hx = c_h * indCurSqOx[1] ;
            }

            if( indCurSqOx[1] < 0) {
                Hx = h * indCurSqOx[1];
            }
        }

        buf_D = d_integUnderRectAng_OneCell( 
                    bv[1],                                  //   -  double Py,
                    uv[1],                                  //   -  double Qy,
                    //
                    mv[0],                                  //   -  double Gx,
                    Hx,                                     //   -  double Hx,
                    //
                      iCurrTL,                           //   -  Index of current time layer.
                    //
                    indCurSqOx,                             //   -  Index of current square by Ox axis.
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    
                    
                    rhoInPrevTL_asV );

        integ += buf_D;
    }

//   Second step: from "mas OX[ indCurSqOx[1] ]" to "rb" by iteration.


    indCurSqOxToCh[0]  =  indCurSqOx[0] +1;
    indCurSqOxToCh[1]  =  indCurSqOxToCh[0] +1;

    for( j = indCurSqOx[0] +1; j< indRB[0] +1; j++ ) {
        //   If this is first cell we should integrate under triangle only.

        if( indCurSqOxToCh[1] > 0 ) {
            Gx = c_h * indCurSqOxToCh[0] ;
            Hx = c_h * indCurSqOxToCh[1];
        }


        if( indCurSqOxToCh[1] <= 0 ) {
            Gx = h * indCurSqOxToCh[0];
            Hx = h * indCurSqOxToCh[1];
        }


        if( j == indRB[0] ) {
            Hx = rb;
        }


        buf_D = d_integUnderRectAng_OneCell( 
                    bv[1],                                  //   -  double Py,
                    uv[1],                                  //   -  double Qy,
                    //
                    Gx,                                     //   -  double Gx,
                    Hx,                                     //   -  double Hx,
                    //
                     
                      iCurrTL,                           //   -  Index of current time layer.
                    //
                    indCurSqOxToCh,                         //   -  Index of current square by Ox axis.
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                   
                    rhoInPrevTL_asV );

        integ += buf_D;

        indCurSqOxToCh[0] +=  1;
        indCurSqOxToCh[1]  =  indCurSqOxToCh[0] +1;
    }

    return integ;
}

__device__ double d_integUnderRigAngTr_BottLeft(  
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,
    double *uv,
    
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
    indCurSqOx[0] = (int)(  (trPC[0] - 1.e-14) /c_h);      //   -  If trPC[0] is in grid edge I want it will be between in the left side of indCurSqOx[1].
    if( (trPC[0] - 1.e-14) <= 0 ) {
        indCurSqOx[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOx[1] = indCurSqOx[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    indRB[0] = indCurSqOx[0];
    indRB[1] = indRB[0] +1;
    indCurSqOy[0] = (int)(  (trPC[1] + 1.e-14) /c_h);      //   -  If trPC[1] is in grid edge I want it will be between indCurSqOx[0] and indCurSqOx[1].
    if( (trPC[1] + 1.e-14) <= 0 ) {
        indCurSqOy[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOy[1] = indCurSqOy[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    if( indCurSqOx[0] >= 0) {
        distOx = trPC[0]  -  c_h * indCurSqOx[0] ;
    }
    if( indCurSqOx[0] < 0 ) {
        distOx = fabs( trPC[0]  -  c_h * indCurSqOx[0] );
    }
    if( indCurSqOy[1] >= 0 ) {
        distOy = c_h * indCurSqOy[1]  -  trPC[1];
    }
    if( indCurSqOy[1] < 0 ) {
        distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
    }
    do {
        //   a. First case.
        if( (distOy /distOx) <= ang ) {
            //   Across with straight line parallel Ox axis.
            wTrPNI = 1;
            if( indCurSqOy[1] >= 0) {
                trPN[1] = c_h * indCurSqOy[1];
            }
            if( indCurSqOy[1] < 0) {
                trPN[1] = c_h * indCurSqOy[1];
            }
            trPN[0] = bv[0] - (trPN[1] - bv[1]) /ang;
        }
        //   b. Second case.
        if( (distOy /distOx) > ang ) {
            //   Across with straight line parallel Oy axis.
            wTrPNI = 2;
            if( indCurSqOx[0] >= 0 ) {
                trPN[0]  =  c_h * indCurSqOx[0];
            }
            if( indCurSqOx[0] < 0 ) {
                trPN[0]  =  c_h * indCurSqOx[0];
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
        buf_D = d_integOfChan_SLLeftSd(  
                     iCurrTL,                           //   -  Index of current time layer.
                    //
                    trPC,  wTrPCI,                          //   -  double *bv,
                    trPN,  wTrPNI,                          //   -  double *uv,
                    //
                    indCurSqOx,                             //   -  Indices where trPC and trPN are.
                    //
                    bv[0], indRB,                           //   -  double rb  =  Right boundary by Ox.
                    //
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                   
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
                distOx = trPC[0]  -  c_h *  indCurSqOx[0] ;
            }
            if( indCurSqOx[0] < 0) {
                distOx = fabs( trPC[0]  -  c_h * indCurSqOx[0] );
            }
            if( indCurSqOy[1] >= 0 ) {
                distOy = c_h *  indCurSqOy[1]  -  trPC[1];
            }
            if( indCurSqOy[1] < 0 ) {
                distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
            }
        }
    } while( !isTrDone );
    return integOfBottTr;
}

__device__ double d_integUnderRigAngTr_BottRight(   
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,
    double *uv,
   
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
    
    double integOfBottTr = 0.;                            //   -  Value which we are computing.
    double buf_D;


    trPC[0] = bv[0];
    trPC[1] = bv[1];
    if(  ( fabs(bv[0] - uv[0]) )  <  1.e-12  ) return fabs(bv[0] - uv[0]);

    ang = (uv[1] - bv[1]) / (uv[0] - bv[0]);
    if(  fabs(ang)  <  1.e-12  ) return fabs(ang);

    indCurSqOx[0] = (int)(  (trPC[0] + 1.e-14) /c_h);      //   -  If trPC[0] is in grid edge I want it will be between in the right side.

    if( (trPC[0] + 1.e-14) <= 0 )  indCurSqOx[0] -= 1;    //   -  The case when "trPC[0]" ia negative.

    indCurSqOx[1] = indCurSqOx[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    indLB[0] = indCurSqOx[0];
    indLB[1] = indLB[0] +1;
    indCurSqOy[0] = (int)(  (trPC[1] + 1.e-14) /c_h);      //   -  If trPC[1] is in grid edge I want it will be in the upper side.
    if( (trPC[1] + 1.e-14) <= 0 ) {
        indCurSqOy[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOy[1] = indCurSqOy[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.

    if( indCurSqOx[1] >=0 ) {
        distOx = fabs( c_h * indCurSqOx[1]  -  trPC[0] );
    }
    if( indCurSqOx[1] < 0 ) {
        distOx = fabs( c_h * indCurSqOx[1]  -  trPC[0] );
    }
    if( indCurSqOy[1] >=0 ) {
        distOy = fabs( c_h * indCurSqOy[1]   -  trPC[1] );
    }
    if( indCurSqOy[1] < 0 ) {
        distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
    }
    do {
        //   a. First case.
        if( (distOy /distOx) <= ang ) {
            //   Across with straight line parallel Ox axis.
            wTrPNI = 1;
            if( indCurSqOy[1] >=0 ) {
                trPN[1] = c_h * indCurSqOy[1];
            }
            if( indCurSqOy[1] < 0 ) {
                trPN[1] = c_h * indCurSqOy[1];
            }
            trPN[0] = bv[0] + (trPN[1] - bv[1]) /ang;
        }
        //   b. Second case.
        if( (distOy /distOx) > ang ) {
            //   Across with straight line parallel Oy axis.
            wTrPNI = 2;
            if( indCurSqOx[1] >= 0 ) {
                trPN[0]  =  c_h * indCurSqOx[1];
            }
            if( indCurSqOx[1]  < 0 ) {
                trPN[0]  =  c_h * indCurSqOx[1];
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
        buf_D = d_integOfChan_SLRightSd( 
                      iCurrTL,                           //   -  Index of current time layer.
                    //
                    trPC,  wTrPCI,                          //   -  double *bv,
                    trPN,  wTrPNI,                          //   -  double *uv,
                    //
                    indCurSqOx,                             //   -  Indices where trPC and trPN are.
                    //
                    bv[0], indLB,                           //   -  double lb  =  Left boundary by Ox.
                    //
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    
                  
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
                distOx = fabs( c_h * indCurSqOx[1]   -  trPC[0] );
            }
            if( indCurSqOx[1] < 0 ) {
                distOx = fabs( c_h * indCurSqOx[1]  -  trPC[0] );
            }
            if( indCurSqOy[1] >=0 ) {
                distOy = fabs( c_h * indCurSqOy[1]   -  trPC[1] );
            }
            if( indCurSqOy[1] < 0 ) {
                distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
            }
        }
    } while( !isTrDone );
    return integOfBottTr;
}

__device__ double d_integUnderBottTr(   
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double * LvBt,                          //   -  Left, Right and Botton vertices of Botton triangle.
    double * RvBt,                          //   -  Left, Right and Botton vertices of Botton triangle.
    double * BvBt,                          //   -  Left, Right and Botton vertices of Botton triangle.
    
    double * rhoInPrevTL_asV,
    int ii, int jj ) // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
{
    double integOfBottTr;
    double buf_D;
    //   Three ways are possible.
    //   1.
    if(  BvBt[0] <= LvBt[0]  ) {
        buf_D = d_integUnderRigAngTr_BottRight(
                       iCurrTL,
                    //
                    BvBt, RvBt,      rhoInPrevTL_asV );
        integOfBottTr = buf_D;
        buf_D = d_integUnderRigAngTr_BottRight(
                     iCurrTL,
                    //
                    BvBt, LvBt,      rhoInPrevTL_asV );
        integOfBottTr = integOfBottTr - buf_D;

//      printf("Bv<Lv: i= %d, j= %d      res= %le",ii,jj,integOfBottTr);  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return integOfBottTr;
    }
    //   2.
    if(  (BvBt[0] > LvBt[0]) && (BvBt[0] < RvBt[0]) ) {

        buf_D = d_integUnderRigAngTr_BottLeft(
               iCurrTL,
                    //
                    BvBt, LvBt,         rhoInPrevTL_asV );
        integOfBottTr = buf_D;

        buf_D = d_integUnderRigAngTr_BottRight(
                     iCurrTL,
                    //
                    BvBt, RvBt,         rhoInPrevTL_asV );
        integOfBottTr = integOfBottTr + buf_D;

//      printf("Bv>Lv & Bv<Rv: i= %d, j= %d      res= %le",ii,jj,integOfBottTr);   // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return integOfBottTr;
    }
    //   3.
    if(  BvBt[0] >= RvBt[0]  ) {

        buf_D = d_integUnderRigAngTr_BottLeft(
               iCurrTL,
                    //
                    BvBt, LvBt,        rhoInPrevTL_asV );
        integOfBottTr = buf_D;
        buf_D = d_integUnderRigAngTr_BottLeft(
                 iCurrTL,
                    //
                    BvBt, RvBt,       rhoInPrevTL_asV );
        integOfBottTr = integOfBottTr - buf_D;

//      printf("Bv>Rv: i= %d, j= %d      res= %le",ii,jj,integOfBottTr);     // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return integOfBottTr;
    }
    return integOfBottTr;
}

__device__ double d_integUnderRigAngTr_UppLeft(  
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,
    double *uv,
     
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
    
    double integOfUppTr = 0.;                             //   -  Value which we are computing.
    double buf_D;
    //   Initial data.
    trPC[0] = bv[0];
    trPC[1] = bv[1];
    if(  ( fabs(bv[0] - uv[0]) )  <  1.e-12  )  return fabs(bv[0] - uv[0]);

    ang = (uv[1] - bv[1]) / (uv[0] - bv[0]);
    if(  fabs(ang)  <  1.e-12  ) return fabs(ang);

    //   The follow equations are quite important.
    indCurSqOx[0] = (int)(  (trPC[0] + 1.e-14) /c_h);      //   -  If trPC[0] is in grid edge I want it will be in the right side.
    if( (trPC[0] + 1.e-14) <= 0 ) {
        indCurSqOx[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOx[1] = indCurSqOx[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    indCurSqOy[0] = (int)(  (trPC[1] + 1.e-14) /c_h);      //   -  If trPC[1] is in grid edge I want it will be in the upper square.
    if( (trPC[1] + 1.e-14) <= 0 ) {
        indCurSqOy[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOy[1] = indCurSqOy[0] +1;
    indRB[0] = (int)(  (uv[0] - 1.e-14) /c_h);             //   -  If uv[0] is in grid edge I want it will be in the left side.
    if( (uv[0] - 1.e-14) <= 0 ) {
        indRB[0] -= 1;     //   -  The case when "trPC[0]" ia negative.
    }
    indRB[1] = indRB[0] +1;
    if( indCurSqOx[1] >= 0) {
        distOx = c_h * indCurSqOx[1]  -  trPC[0];
    }
    if( indCurSqOx[1] < 0) {
        distOx = fabs( c_h * indCurSqOx[1]  -  trPC[0] );
    }
    if( indCurSqOy[1] >= 0 ) {
        distOy = c_h * indCurSqOy[1]  -  trPC[1];
    }
    if( indCurSqOy[1] < 0 ) {
        distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
    }
    do {
        //   a. First case.
        if( (distOy /distOx) <= ang ) {
            //   Across with straight line parallel Ox axis.
            wTrPNI = 1;
            if( indCurSqOy[1] >= 0 ) {
                trPN[1] = c_h * indCurSqOy[1];
            }
            if( indCurSqOy[1] < 0 ) {
                trPN[1] = c_h * indCurSqOy[1];
            }
            trPN[0] = bv[0] + (trPN[1] - bv[1]) /ang;
        }
        //   b. Second case.
        if( (distOy /distOx) > ang ) {
            //   Across with straight line parallel Oy axis.
            wTrPNI = 2;
            if( indCurSqOx[1] >= 0 ) {
                trPN[0]  =  c_h * indCurSqOx[1];
            }
            if( indCurSqOx[1] < 0 ) {
                trPN[0]  =  c_h * indCurSqOx[1];
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
        buf_D = d_integOfChan_SLLeftSd(  
                     iCurrTL,                           //   -  Index of current time layer.
                    //
                    trPC,  wTrPCI,                          //   -  double *bv,
                    trPN,  wTrPNI,                          //   -  double *uv,
                    //
                    indCurSqOx,                             //   -  Indices where trPC and trPN are.
                    //
                    uv[0], indRB,                           //   -  double rb  =  Right boundary by Ox.
                    //
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    
                    
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
                distOx = fabs( c_h * indCurSqOx[1]  -  trPC[0] );
            }
            if( indCurSqOx[1] < 0) {
                distOx = fabs( c_h * indCurSqOx[1]  -  trPC[0] );
            }
            if( indCurSqOy[1] >= 0 ) {
                distOy = fabs( c_h * indCurSqOy[1] -  trPC[1] );
            }
            if( indCurSqOy[1] < 0 ) {
                distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
            }
        }
    } while( !isTrDone );
    return integOfUppTr;
}

__device__ double d_integUnderRigAngTr_UppRight(  
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double *bv,
    double *uv, 
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
    indCurSqOx[0] = (int)(  (trPC[0] - 1.e-14) /c_h);      //   -  If trPC[0] is in grid edge I want it will be between in the left side.
    if( (trPC[0] - 1.e-14) <= 0 ) {
        indCurSqOx[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOx[1] = indCurSqOx[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    indLB[0] = (int)( (uv[0] + 1.e-14) /c_h);
    if( (uv[0] + 1.e-14) <=0 ) {
        indLB[0] -= 1;     //   -  The case when "trPC[0]" ia negative.
    }
    indLB[1] = indLB[0] +1;
    indCurSqOy[0] = (int)(  (trPC[1] + 1.e-14) /c_h);      //   -  If trPC[1] is in grid edge I want it will be in the upper side.
    if( (trPC[1] + 1.e-14) <= 0 ) {
        indCurSqOy[0] -= 1;    //   -  The case when "trPC[0]" ia negative.
    }
    indCurSqOy[1] = indCurSqOy[0] +1;                     //   -  It's important only in rare case then trPC is in grid edge.
    if( indCurSqOx[0] >= 0 ) {
        distOx = fabs( trPC[0]  -  c_h * indCurSqOx[0] );
    }
    if( indCurSqOx[0] < 0 ) {
        distOx = fabs( trPC[0]  -  c_h * indCurSqOx[0] );
    }
    if( indCurSqOy[1] >= 0 ) {
        distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
    }
    if( indCurSqOy[1] < 0 ) {
        distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
    }
    do {
        //   a. First case.
        if( (distOy /distOx) <= ang ) {
            //   Across with straight line parallel Ox axis.
            wTrPNI = 1;
            if( indCurSqOy[1] >= 0 ) {
                trPN[1] = c_h * indCurSqOy[1];
            }
            if( indCurSqOy[1] < 0 ) {
                trPN[1] = c_h * indCurSqOy[1];
            }
            trPN[0] = bv[0] - (trPN[1] - bv[1]) /ang;
        }
        //   b. Second case.
        if( (distOy /distOx) > ang ) {
            //   Across with straight line parallel Oy axis.
            wTrPNI = 2;
            if( indCurSqOx[0] >= 0 ) {
                trPN[0]  =  c_h * indCurSqOx[0];
            }
            if( indCurSqOx[0] < 0 ) {
                trPN[0]  =  c_h * indCurSqOx[0];
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
        buf_D = d_integOfChan_SLRightSd( 
                     iCurrTL,                           //   -  Index of current time layer.
                    //
                    trPC,  wTrPCI,                          //   -  double *bv,
                    trPN,  wTrPNI,                          //   -  double *uv,
                    //
                    indCurSqOx,                             //   -  Indices where trPC and trPN are.
                    //
                    uv[0], indLB,                           //   -  double lb  =  Left boundary by Ox.
                    //
                    indCurSqOy,                             //   -  Index of current square by Oy axis.
                    
                    
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
                distOx = fabs( trPC[0]  - c_h * indCurSqOx[0] );
            }
            if( indCurSqOx[0] < 0 ) {
                distOx = fabs( trPC[0]  -  c_h * indCurSqOx[0] );
            }
            if( indCurSqOy[1] >= 0 ) {
                distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
            }
            if( indCurSqOy[1] < 0 ) {
                distOy = fabs( c_h * indCurSqOy[1]  -  trPC[1] );
            }
        }
    } while(!isTrDone);
    return integOfUppTr;
}

__device__ double d_integUnderUpperTr(  
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double * LvUt,                          //   -  Left, Right and Upper vertices of Upper triangle.
    double * RvUt,                          //   -  Left, Right and Upper vertices of Upper triangle.
    double * UvUt,                          //   -  Left, Right and Upper vertices of Upper triangle.
   
    double * rhoInPrevTL_asV)
{
    double integOfUppTr;
    double buf_D;
    //   Three ways are possible.
    //   1.
    if(  UvUt[0] <= LvUt[0]  ) {
        buf_D = d_integUnderRigAngTr_UppRight(
                       iCurrTL,
                    //
                    RvUt, UvUt,         rhoInPrevTL_asV );
        integOfUppTr = buf_D;
        buf_D = d_integUnderRigAngTr_UppRight(
                      iCurrTL,
                    //
                    LvUt, UvUt,        rhoInPrevTL_asV );
        integOfUppTr = integOfUppTr - buf_D;
        return integOfUppTr;
    }
    //   2.
    if(  (UvUt[0] > LvUt[0]) && (UvUt[0] < RvUt[0]) ) {
        buf_D = d_integUnderRigAngTr_UppLeft(
                  iCurrTL,
                    //
                    LvUt, UvUt,     rhoInPrevTL_asV );
        integOfUppTr = buf_D;

        buf_D = d_integUnderRigAngTr_UppRight(
                      iCurrTL,
                    //
                    RvUt, UvUt,       rhoInPrevTL_asV );
        integOfUppTr = integOfUppTr + buf_D;
        return integOfUppTr;
    }
    //   3.
    if(  UvUt[0] >= RvUt[0]  ) {
        buf_D = d_integUnderRigAngTr_UppLeft(
                      iCurrTL,
                    //
                    LvUt, UvUt,     rhoInPrevTL_asV );
        integOfUppTr = buf_D;
        buf_D = d_integUnderRigAngTr_UppLeft(
                 iCurrTL,
                    //
                    RvUt, UvUt,      rhoInPrevTL_asV );
        integOfUppTr = integOfUppTr - buf_D;
        return integOfUppTr;
    }
    return integOfUppTr;
}

__device__ double d_integUnderUnunifTr(  
    int iCurrTL,                            //   -  Index of current time layer.
    //
    double * firVer,                        //   -  First vertex of triangle.
    double * secVer,                        //   -  Second vertex of triangle.
    double * thiVer,                        //   -  Third vertex of triangle.
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
    uv[1] = 0;                                     //   -  The minimum possible value.
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

//  printf("i= %d, j= %d : ap[0]= %le      mv[0]= %le \n",ii,jj, ap[0], mv[0]); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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
                              iCurrTL,                           //   -  Index of current time layer.
                            //
                            LvBt, RvBt, BvBt,                       //   -  Left, Right and Botton vertices of Botton triangle.
                            
                            //
                            rhoInPrevTL_asV,
                            ii, jj ); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integ = integOfBottTr;
 
        //   Left, Right and Upper vertices of Upper triangle.
        LvUt[0]  =  mv[0];
        LvUt[1]  =  mv[1];
        RvUt[0]  =  ap[0];
        RvUt[1]  =  ap[1];
        UvUt[0]  =  uv[0];
        UvUt[1]  =  uv[1];
    
        integOfUppTr = d_integUnderUpperTr( 
                           iCurrTL,                           //   -  Index of current time layer.
                           //
                           LvUt, RvUt, UvUt,                       //   -  Left, Right and Botton vertices of Upper triangle.
                           
                           //
                           rhoInPrevTL_asV);

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
                           iCurrTL,                           //   -  Index of current time layer.
                            //
                            LvBt, RvBt, BvBt,                       //   -  Left, Right and Botton vertices of Botton triangle.
                             
                            //
                            rhoInPrevTL_asV,
                            ii, jj ); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integ = integOfBottTr; 
        //   Left, Right and Upper vertices of Upper triangle.
        LvUt[0]  =  ap[0];
        LvUt[1]  =  ap[1];
        RvUt[0]  =  mv[0];
        RvUt[1]  =  mv[1];
        UvUt[0]  =  uv[0];
        UvUt[1]  =  uv[1];
        integOfUppTr = d_integUnderUpperTr( 
                          iCurrTL,                           //   -  Index of current time layer.
                           //
                           LvUt, RvUt, UvUt,                       //   -  Left, Right and Botton vertices of Upper triangle.
                           
                           rhoInPrevTL_asV );
        return integ + integOfUppTr;
    }
    return integ;
}

__device__ double d_f_function(const int current_tl, const int i, const int j)
{ 
    double x  =  c_h * i ;
    double y  =  c_h * j ;
    double arg_v  =  (x - c_lb) * (x - c_rb) * (1.+c_tau*current_tl) /10. * (y - c_ub) * (y - c_bb);
    double rho, dRhoDT, dRhoDX, dRhoDY;
    double u, duDX;
    double v, dvDY;
    rho  =  d_analytSolut(c_tau*current_tl, x, y );
    dRhoDT  =  x * y * cos( c_tau*current_tl*x*y );
    dRhoDX  =  c_tau*current_tl * y * cos( c_tau*current_tl*x*y );
    dRhoDY  =  c_tau*current_tl * x * cos( c_tau*current_tl*x*y );
    u  =  d_u_function(c_tau*current_tl, x, y );
    duDX  = -c_b * y * (1.-y)  /  ( 1.  +  x * x );
    v  =  d_v_function(c_tau*current_tl, x, y );
    dvDY  =  (x - c_lb) * (x - c_rb) * (1.+c_tau*current_tl) /10. * (y - c_bb + y - c_ub);
    dvDY  =  dvDY  /  ( 1.  +  arg_v * arg_v );
    double res = dRhoDT   +   rho * duDX   +   u * dRhoDX   +   rho * dvDY   +   v * dRhoDY;
    return res;
}

__device__ double space_volume_in_prev_tl(double* prev_result, int current_tl, int i, int j)
{
    double first1[2]; double second1[2]; double third1[2];
    double first2[2]; double second2[2]; double third2[2];

    double x, y;
    double c_tau_to_current_tl = (1. + current_tl * c_tau) / 10.;

    // A
    x = (c_h*(i - 1) + c_h*i) / 2.;
    y = (c_h*(j - 1) + c_h*j) / 2.;
    first1[0] = first2[0] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
    first1[1] = first2[1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_tl * (y - c_ub) * (y - c_bb));
    // B
    x = (c_h*(i + 1) + c_h*i) / 2.;
    second1[0] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
    second1[1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_tl * (y - c_ub) * (y - c_bb));
    // C
    y = (c_h*(j + 1) + c_h*j) / 2.;
    third1[0] = third2[0] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
    third1[1] = third2[1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_tl * (y - c_ub) * (y - c_bb));
    // D 
    x = (c_h*(i - 1) + c_h*i) / 2.;
    second2[0] = x - c_tau_b * y * (1. - y) * (c_pi_half + atan(-x));
    second2[1] = y - c_tau * atan((x - c_lb) * (x - c_rb) * c_tau_to_current_tl * (y - c_ub) * (y - c_bb));


    double buf_D = d_integUnderUnunifTr( 
                    current_tl,
                    first1, second1, third1, 
                    prev_result,
                    i, j);

    return buf_D + d_integUnderUnunifTr( 
           current_tl,       
           first2, second2, third2,                   
           prev_result,
           i, j );
}

__global__ void kernel(double* prev_result, double* result, int current_tl)
{
    for (int opt = blockIdx.x * blockDim.x + threadIdx.x; opt < c_n; opt += blockDim.x * gridDim.x)
    {
        int i = opt % (c_x_length + 1);
        int j = opt / (c_y_length + 1);
       
        // расчет границы
        if (j == 0)  // bottom bound
        {
            result[ opt ]  = 1.1  +  sin( c_tau_to_h * current_tl * j * c_bb );
        }
        else if (i == 0) // left bound
        {
            result[ opt ] = 1.1  +  sin( c_tau_to_h * current_tl * i * c_lb );
        }
        else if (j == c_y_length) // upper bound
        { 
            result[ opt ] = 1.1  +  sin( c_tau_to_h * current_tl * i * c_ub );
        }
        else if (i == c_x_length) // right bound
        { 
            result[ opt ] = 1.1  +  sin(  c_tau_to_h * current_tl * j * c_rb );
        }
        else if (i > 0 && j > 0 && j != c_x_length && i != c_x_length)
        {        
            result[ opt ] = space_volume_in_prev_tl(prev_result, current_tl, i, j);
            double t = space_volume_in_prev_tl(prev_result, current_tl, i, j) / c_h;
            t = t / c_h;
            if (opt == 16)
            { 
               printf("gpu result[%d] = %le\n", opt, t);
               printf("c_h = %le\n", c_h);
               printf("i = %d, j = %d\n", i, j);
	       printf("c_h*i = %le c_h*j = %le\n", c_h*i, c_h*j);
               printf("current tl = %d\n", current_tl);
               double tmp =  d_analytSolut(current_tl, c_h*i, c_h*j);
               printf("gpu analytSolut = %le\n", tmp);
            }
            result[ opt ] = t;
            result[ opt ] += c_tau * d_f_function(current_tl, i, j);
            if (opt == 16)
            {
               printf("gpu result[%d] = %le\n", opt, result[opt]);
            } 
        }
    }
}

double* init_rho(ComputeParameters *p)
{
    double *rhoInPrevTL_asV;

    rhoInPrevTL_asV = new double [ p->size ];
    //   Initial data of rho.
    for( int k = 0; k <= p->x_size; k++ ) {
        for( int j = 0; j <= p->y_size; j++ ) {
            rhoInPrevTL_asV[ (p->x_size+1)*k + j ]  =  1.1  +  sin( 0.* p->x[ k ] * p->y[ j ]);
        }
    }
    return rhoInPrevTL_asV;
}

float solve_at_gpu(ComputeParameters *p, bool tl1)
{
    assert(p != NULL);
    assert(p->result != NULL);
    //const int gridSize = 256;
    //const int blockSize =  512;
    const int gridSize = 1;
    const int blockSize =  1;
    size_t n(0);
    int temp_i(0);
    double temp_d(0);
    double *result = NULL, *prev_result = NULL;
    n = p->get_real_matrix_size();
    int size = sizeof(double)*n;
    double *rhoInPrevTL_asV = init_rho(p);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMemcpyToSymbol(c_tau, &p->tau, sizeof(double));
    cudaMemcpyToSymbol(c_lb, &p->lb, sizeof(double));
    cudaMemcpyToSymbol(c_b, &p->b, sizeof(double));
    cudaMemcpyToSymbol(c_rb, &p->rb, sizeof(double));
    cudaMemcpyToSymbol(c_bb, &p->bb, sizeof(double));
    cudaMemcpyToSymbol(c_ub, &p->ub, sizeof(double));
    cudaMemcpyToSymbol(c_n, &n, sizeof(int));
   
    temp_i = p->x_size;
    cudaMemcpyToSymbol(c_x_length, &temp_i, sizeof(int));
    temp_i = p->y_size;
    cudaMemcpyToSymbol(c_y_length, &temp_i, sizeof(int));
    temp_d = 1. / (p->x_size);
    cudaMemcpyToSymbol(c_h, &temp_d, sizeof(double));

    temp_d = p->tau / (p->x_size);
    cudaMemcpyToSymbol(c_tau_to_h, &temp_d, sizeof(double));

    temp_d = p->b * p->tau;
    cudaMemcpyToSymbol(c_tau_b, &temp_d, sizeof(double));

    temp_d = C_pi_device / 2.;
    cudaMemcpyToSymbol(c_pi_half, &temp_d, sizeof(double));
    
    checkCuda(cudaMalloc((void**)&(result), size) );
    checkCuda(cudaMalloc((void**)&(prev_result), size) );
    cudaMemcpy(prev_result, rhoInPrevTL_asV, size, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start, 0);   

    if (tl1 == true)
    {
        kernel<<<gridSize, blockSize>>>(prev_result, result, 1);
        cudaMemcpy(p->result, result, size, cudaMemcpyDeviceToHost);
    }
    else
    {
        int tl = 0;
        int tempTl = p->t_count - 1;  
        while(tl < tempTl)
        {
            kernel<<<gridSize, blockSize>>>(prev_result, result, tl + 1);
            kernel<<<gridSize, blockSize>>>(result, prev_result, tl + 2);         
            tl += 2;            
        }  
        cudaMemcpy(p->result, prev_result, size, cudaMemcpyDeviceToHost);
     }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaFree(result);
    cudaFree(prev_result);
    cudaDeviceReset();
    delete[] rhoInPrevTL_asV;
    return time;
}

