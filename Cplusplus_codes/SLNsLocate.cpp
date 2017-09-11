
#include "SLNsLocate.h"
#include <iostream> 
#include <fstream>

SLNsLocate::SLNsLocate(){}

SLNsLocate::~SLNsLocate(){}

void SLNsLocate::sortRect( vector<cv::Rect> &rects )
{
	vector<cv::Rect> trects;
	cv::Rect tr;
	trects = rects;
	for ( size_t i = 0; i < trects.size(); i++ )
	{
		for ( size_t j = i+1; j < trects.size(); j++ )
		{
			if ( trects[j].x < trects[i].x )
			{
				tr = trects[i];
				trects[i] = trects[j];
				trects[j] = tr;
			}
		}
	}
	rects = trects;
}

void SLNsLocate::sortRectY( vector<cv::Rect> &rects )
{
	vector<cv::Rect> trects;
	cv::Rect tr;
	trects = rects;
	for ( size_t i = 0; i < trects.size(); i++ )
	{
		for ( size_t j = i+1; j < trects.size(); j++ )
		{
			if ( trects[j].y < trects[i].y )
			{
				tr = trects[i];
				trects[i] = trects[j];
				trects[j] = tr;
			}
		}
	}
	rects = trects;
}
string SLNsLocate::intTostring(int i)
{
	string c;
	ostringstream oss;

	oss << i;

	c = oss.str();

	return c;
}

Mat SLNsLocate::labHist(const Mat& src)
{  
    Mat lab;
    cvtColor(src,lab,CV_BGR2Lab);  
  
    int lbins = 32;  
    int abins = 32;  
    int bbins = 32;  
    int histSize[] = { lbins , abins , bbins};  
  
    float lranges [ ] ={0,255};
    float aranges [ ] ={0,255};
    float branges [ ] ={0,255};
    const float* ranges [ ]={lranges ,aranges , branges};  
  
    Mat hist3D,hist3dNormal;  
    Mat hist =Mat(lbins*abins*bbins,1,CV_64FC1);  
  
    const int channels [ ]={0,1,2};  
    calcHist(&lab,1,channels,Mat(),hist3D,3,histSize,ranges,true,false);

    normalize(hist3D,hist3dNormal,1,0,CV_L1,CV_64F);  
      

    int row = 0;
    for(int l = 0; l < lbins; l++)  
    {  
        for(int a = 0;a < abins; a++)  
        {  
            for(int b = 0;b < bbins;b++)  
            {  
                hist.at<double>(row,0)=*((double*)(hist3dNormal.data+l*hist3dNormal.step[0]+a*hist3dNormal.step[1]+b*hist3dNormal.step[2]));  
                row++;  
            }  
        }  
    }

    return hist;  
} 

double SLNsLocate::chiSquareDist(const Mat & hist1,const Mat & hist2)
{  
    int rows = hist1.rows;  
    double sum = 0.0;  
    double d1,d2;  
    for(int r = 0;r < rows ;r++)  
    {  
        d1 = hist1.at<double>(r,0);  
        d2 = hist2.at<double>(r,0);  
        if(!( d1 ==0 && d2 == 0) )
            sum += 0.5*pow( d1 - d2,2)/(d1+d2);  
    }  
    return sum;  
}  

int SLNsLocate::getHopCount(uchar i)
{
    int a[8]={0};
    int k=7;
    int cnt=0;
    while(i)
    {
        a[k]=i&1;
        i>>=1;
        --k;
    }
    for(int k=0;k<8;++k)
    {
        if(a[k]!=a[k+1==8?0:k+1])
        {
            ++cnt;
        }
    }
    return cnt;
}

void SLNsLocate::lbp59table(uchar* table)
{
    memset(table,0,256);
    uchar temp=1;
    for(int i=0;i<256;++i)
    {
        if(getHopCount(i)<=2)
        {
            table[i]=temp;
            temp++;
        }
    }
}

Mat SLNsLocate::histUniformLBP(Mat &image)
{
    uchar table[256];
    lbp59table(table);

	Mat ggray;
	cv::cvtColor(image, ggray, CV_BGR2GRAY);

    Mat result;
    result.create(Size(ggray.cols, ggray.rows), ggray.type() );

    for(int y = 1; y < ggray.rows-1; y ++)
    {
        for(int x = 1; x < ggray.cols-1; x++)
        {
            uchar neighbor[8] = {0};
            neighbor[0] = ggray.at<uchar>(y-1, x-1);
            neighbor[1] = ggray.at<uchar>(y-1, x);
            neighbor[2] = ggray.at<uchar>(y-1, x+1);
            neighbor[3] = ggray.at<uchar>(y, x+1);
            neighbor[4] = ggray.at<uchar>(y+1, x+1);
            neighbor[5] = ggray.at<uchar>(y+1, x);
            neighbor[6] = ggray.at<uchar>(y+1, x-1);
            neighbor[7] = ggray.at<uchar>(y, x-1);
            uchar center = ggray.at<uchar>(y, x);
            uchar temp = 0;
            for(int k = 0; k < 8; k++)
            {
                temp += (neighbor[k] >= center)* (1<<k);  // 计算LBP的值
            }
            result.at<uchar>(y,x) = table[temp];   //  降为59维空间
        }
    }

    int bins = 59;
    const int histSize[1]={bins};
    float lranges [ ] ={0,255};
    const float* ranges [ ]={lranges};

    Mat hist,histNormal;
    hist = Mat(bins,1,CV_64FC1);

    const int channels[1] = {0};
    calcHist(&result,1,channels,Mat(),hist,1,histSize,ranges,true,false);

    normalize(hist,histNormal,1,0,CV_L1,CV_64FC1);


    return histNormal;
}

vector<cv::Rect> SLNsLocate::SLNsFiltering(cv::Mat srcColor, vector<cv::Rect> fineRects, SPPARA &sppa)
{

		Mat srcOrignal;
		srcColor.copyTo(srcOrignal);
		vector<cv::Rect> LastFinal;
		vector<cv::Rect> LastPlateRects = fineRects;

		if(fineRects.size()<1)
			return LastFinal;

		for (int i = 0; i < static_cast<int> (LastPlateRects.size()); i++)
		{
			Rect curFinal;
			Mat curLast;
			Rect curRect = LastPlateRects[i];

			Mat binSrc,gray;
			srcOrignal(curRect).copyTo(binSrc);

			if( binSrc.channels() != 1 )
				cv::cvtColor( binSrc, gray, CV_BGR2GRAY );
			else
				binSrc.copyTo( gray );


			vector<vector<Point> > contours;

			int _min_area = 0.01*curRect.area();
			int _max_area = 0.5*curRect.area();
			cv::MSER ms(5, _min_area, _max_area);

			ms(gray, contours, Mat());


			 vector<cv::Rect> RectsMser;
		     Mat result3 = Mat::zeros(gray.size(),CV_8UC3);

			 double wBr = 0.05;
			 double wBrMax = 0.4;
		     double hBr = 0.25;
			 double aBr = 0.025;
			 double aBrMax = 0.5;

			 for (int i = 0; i < static_cast<int> (contours.size()); i++)
			 {
					Rect tr = cv::boundingRect(cv::Mat(contours[i]));

					double wRatio = tr.width / (double)curRect.width;
					double hRatio = tr.height / (double)curRect.height;
					double areaRatio = tr.area() / (double)curRect.area();

					if((wRatio < wBr) || (wRatio > wBrMax))
					{
						contours.erase(contours.begin()+i);
						i--;
						continue;
					}
					else if(hRatio < hBr)
					{
						contours.erase(contours.begin()+i);
						i--;
						continue;
					}
					else if( (areaRatio < aBr) || (areaRatio > aBrMax) )
					{
						contours.erase(contours.begin()+i);
						i--;
						continue;
					}
					cv::rectangle(result3, tr, cv::Scalar(255,255,255),1,1,0);
					RectsMser.push_back(tr);
			 }

	    double mgS = 0.3;

        vector<cv::Rect> RectsFinal;
		if(RectsMser.size()>0)
		{
             sortRect(RectsMser);
             RectsFinal.push_back(RectsMser[0]);

			 for(int i = 1; i < static_cast<int> (RectsMser.size() ); i++)
			 {
				 Rect ect = RectsMser[i] & RectsFinal[RectsFinal.size()-1];
				 if (ect.area() > mgS*RectsMser[i].area())
				 {
					 Rect ectN = RectsMser[i] | RectsFinal[RectsFinal.size()-1];
					 RectsFinal.pop_back();
					 RectsFinal.push_back(ectN);
				 }
				 else
					 RectsFinal.push_back(RectsMser[i]);
			 }
		}

		if (static_cast<int> (RectsFinal.size()) < sppa.SF_RectsNum_min || static_cast<int> (RectsFinal.size()) > sppa.SF_RectsNum_max)  //少于2个的删除
						continue;

		 vector<cv::Rect> RectsS = RectsFinal;
		 double labSumAll=0,lbpSumAll=0;

		 for(int i = 0; i < static_cast<int> (RectsS.size() ); i++)
		 {
           Mat base = binSrc(RectsS[i]);
           //imwrite("base.png",base);

			 for(int j = i+1; j < static_cast<int> (RectsS.size() ); j++)
			 {
				 Mat cur = binSrc(RectsS[j]);
			//	 imwrite("cur.png",cur);
				 double labSum = chiSquareDist(labHist(base), labHist(cur));
				 double lbpSum = chiSquareDist(histUniformLBP(base), histUniformLBP(cur));

				 labSumAll += labSum;
				 lbpSumAll += lbpSum;
			 }

		 }
		 int num = (RectsS.size()-1)*RectsS.size()/2;
		 if (num == 0)
			 continue;
		 labSumAll /= double(num);
		 lbpSumAll /= double(num);

		// cout << labSumAll << endl;
		// cout << lbpSumAll << endl;

		 if (lbpSumAll > sppa.SF_lbpSumAll || labSumAll > sppa.SF_labSumAll)
			 continue;


         LastFinal.push_back(curRect);
 }
	return LastFinal;
}

vector<cv::Rect> SLNsLocate::rectsMerge(cv::Mat src, vector<cv::Rect> coarseRects, SPPARA &sppa)
{
	vector<vector<cv::Rect> > MergedV;
	vector<cv::Rect> sortRects, MergedRects;
	sortRects = coarseRects;
	sortRect(sortRects);

	if(coarseRects.size()<1)
		return MergedRects;

	if(sortRects.size()>1)
	{
		for(int j=0; j < static_cast<int> (sortRects.size()); j++)
		{
			MergedRects.clear();

			if(j == 0)
			{
				MergedRects.push_back(sortRects[j]);
				MergedV.push_back(MergedRects);
				continue;
			}

			Rect curRect = sortRects[j];
			bool ifMerged = false;

			for(int i=0; i < static_cast<int> (MergedV.size()); i++)
			{
				Rect curM = MergedV[i][MergedV[i].size()-1];

				int curWidth = curRect.x - curM.x - curM.width;
			    if (curWidth <= sppa.RM_w || ((curM & curRect).area() > 0))
			    {
			    	MergedV[i].push_back(curRect);
			    	ifMerged = true;
			    }
			}

			if(!ifMerged)
			{
				MergedRects.push_back(curRect);
				MergedV.push_back(MergedRects);
				continue;
			}

		}
	}
	else
	{
		MergedV.push_back(sortRects);
	}


	vector<vector<vector<cv::Rect> > > MergedH;
	for(int j=0; j < static_cast<int> (MergedV.size()); j++)
	{
			vector<vector<cv::Rect> > curMergedH;

			vector<cv::Rect> curSort = MergedV[j];

			if(curSort.size() > 1)
			{
			sortRectY(curSort);

			for(int v=0; v < static_cast<int> (curSort.size()); v++)
			{
				MergedRects.clear();

				if(v == 0)
				{
					MergedRects.push_back(curSort[v]);
					curMergedH.push_back(MergedRects);
					continue;
				}

			Rect curRect = curSort[v];
			bool ifMerged = false;

			for(int i=0; i < static_cast<int> (curMergedH.size()); i++)
			{
				Rect curM = curMergedH[i][curMergedH[i].size()-1];
				int curH = curRect.y - curM.y - curM.height;
			    if (curH <= sppa.RM_h || ((curM & curRect).area() > 0))
			    {
			    	curMergedH[i].push_back(curRect);
			    	ifMerged = true;
			    }
			}

			if(!ifMerged)
			{
				MergedRects.push_back(curSort[v]);
				curMergedH.push_back(MergedRects);
				continue;
			}

		}
		}
		else
		{
			curMergedH.push_back(curSort);
		}
			MergedH.push_back(curMergedH);
	}


	vector<cv::Rect>  finalRects;



	double wBr = sppa.RM_wBr;
	double hBr = sppa.RM_hBr;

	for(int j=0; j < static_cast<int> (MergedH.size()); j++)
	{
		for(int i=0; i < static_cast<int> (MergedH[j].size()); i++)
		{
			Rect rectF;

			for(int p=0; p < static_cast<int> (MergedH[j][i].size()); p++)
			{
				if(p==0)
					rectF = MergedH[j][i][p];
				else
					rectF = rectF | MergedH[j][i][p];
			}

				double wRatio = rectF.width / (double)src.cols;
				double hRatio = rectF.height / (double)src.rows;

				if((wRatio >= wBr) || (hRatio >= hBr))
					continue;

			finalRects.push_back(rectF);
		}
	}

		return finalRects;
}
