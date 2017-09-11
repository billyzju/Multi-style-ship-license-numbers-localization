


#include "paraDefine.h"
#include <iostream>
#include <vector>
#include <malloc.h>
#include <math.h>

#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

class SLNsLocate
{
public:
	SLNsLocate();
	~SLNsLocate();

	void sortRect( vector<cv::Rect> &rects );
	string intTostring(int i);
	Mat labHist(const Mat& src);
	double chiSquareDist(const Mat & hist1,const Mat & hist2);

	 int getHopCount(uchar i);
	 void lbp59table(uchar *table);
	 cv::Mat histUniformLBP(Mat &image);
	 void sortRectY( vector<cv::Rect> &rects );
	 vector<cv::Rect> rectsMerge(cv::Mat src, vector<cv::Rect> coarseRects, SPPARA &sppa);
	 vector<cv::Rect> SLNsFiltering(cv::Mat srcColor, vector<cv::Rect> fineRects, SPPARA &sppa);
};
