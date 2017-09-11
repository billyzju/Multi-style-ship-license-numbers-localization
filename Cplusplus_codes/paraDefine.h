
#include <cv.h>
#include <highgui.h>


#ifndef PARAM_H
#define PARAM_H

#define debug 0
#define SKIP 0
#define STOP 950

typedef struct param{
	std::string filename;
	int id;
}PARAM;


typedef struct superPara{
	int RC_w;
	int RC_h;

    double SF_labSumAll;
    double SF_lbpSumAll;
    int SF_RectsNum_min;
    int SF_RectsNum_max;

	int RM_w;
	int RM_h;
	double RM_hBr;
	double RM_wBr;
	int id;
}SPPARA;

#endif
