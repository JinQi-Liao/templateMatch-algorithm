#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

#define WINDOW_NAME1 "[Original Window]"        
#define WINDOW_NAME2 "[Effect Window]"        


//Template matching algorithm based on FFT technique which is realized by calling matchTemplate function in Opencv3.
cv::Point matchprocessing(cv::Mat srcImage, cv::Mat templateImage, int method = TM_CCOEFF_NORMED)
{
	//Set matrix size and type of resultImage
	cv::Mat resultImage;
	int resultImage_rows = srcImage.rows - templateImage.rows + 1;
	int resultImage_cols = srcImage.cols - templateImage.cols + 1;
	resultImage.create(resultImage_rows, resultImage_cols, CV_32FC1);

	//Call matchTemplate function
	matchTemplate(srcImage, templateImage, resultImage, method);

	//Normalization
	normalize(resultImage, resultImage, 0, 1, NORM_MINMAX, -1, cv::Mat());

	//Call minMaxLoc function to locate best match point
	double minVaule;
	double maxVaule;
	cv::Point minLocation;
	cv::Point maxLocation;
	minMaxLoc(resultImage, &minVaule, &maxVaule, &minLocation, &maxLocation);

	cv::Point matchLocation;
	double matchVaule;
	if (method == TM_SQDIFF || method == TM_SQDIFF_NORMED)
	{
		matchLocation = minLocation;
		matchVaule = minVaule;
	}
	else
	{
		matchLocation = maxLocation;
		matchVaule = maxVaule;
	}

	//return best match point.
	return matchLocation;
}

int main()
{
	double averTime = 0;
	for (int i = 0; i < 32; i++)
	{
		std::cout << " The " << i + 1 << " test£º" << endl;

         double time0 = static_cast<double>(getTickCount());

		Mat srcImage = imread("6-13.bmp", 0);
		Mat templateImage = imread("T6.bmp", 0);
		
		Point bestPoint = matchprocessing(srcImage, templateImage);

		time0 = ((double)getTickCount() - time0) / getTickFrequency();
		if (i > 1)
		{
			averTime += time0;
		}
		std::cout << bestPoint << std::endl << endl;
	}

	std::cout << "The average time: " <<1000* averTime / 30 << endl;
	waitKey();
	return 0;
}