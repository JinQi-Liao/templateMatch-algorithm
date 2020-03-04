#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define WINDOW_NAME1 "[Original Window]"
#define WINDOW_NAME2 "[Effect Window]"

using namespace std;
using namespace cv;

// 1. Calculate the best sampling factor, the principle is to calculate the minimum value of the corresponding function
double sampleFactor(const Mat & srcImage, const Mat & templateImage)
{
	double temp = (srcImage.rows - templateImage.rows + 1) * (srcImage.cols - templateImage.cols + 1);
	double factor = std::pow(temp / 8, 1.0 / 6);
	return factor;
}


//2. Downsample image which means the size of image is reduced and the scale ratios in the x and y direction are the same 
cv::Mat subSample(const Mat & srcImage, double factor)
{
	Mat resultImage;

	//Call resize() function
	cv::resize(srcImage, resultImage, Size(), 1 / factor, 1 / factor, INTER_AREA);

	return resultImage;
}


//3. Obtain the field of 2*scale*factor X 2*scale * factor with (x, y) as the center point and return the top-left vertet of the field. 
cv::Point crop_Image(const Mat & srcImage, const Mat & templateImage, Mat & cropImage, int x, int y, double factor, int scale)
{
	
	//The first is coordinate mapping, which maps the coordinates of the sampled submap to the source image.
	int srcx = int(x * factor);
	int srcy = int(y * factor);


	
	//Calculate the coordinates of the upper left corner and perform the upper left corner boundary processing
	int leftTopx = srcx - int(scale * factor);
	int leftTopy = srcy - int(scale * factor);

	//Calculate the coordinates of the bottom right corner and perform the boundary processing of the bottom right corner
	int rightBottomx = srcx + int(scale * factor) + templateImage.cols;
	int rightBottomy = srcy + int(scale * factor) + templateImage.rows;


	//Boundary processing, but the boundary condition is either the left or top boundary is exceeded, or the right or bottom boundary is exceeded, but the two will not occur at the same time.
	//This kind of boundary processing can ensure that the size of the segmented image must be larger than the size of the template, and the subsequent matching process will not go wrong.
	if (leftTopx < 0)//If it exceeds the left boundary, the whole is translated to the right
	{
		rightBottomx -= leftTopx;
		leftTopx = 0;
	}

	if (leftTopy < 0)//If it exceeds the upper boundary, the whole pans down
	{
		rightBottomy -= leftTopy;
		leftTopy = 0;
	}

	if (rightBottomx > srcImage.cols - 1)//If the right boundary is exceeded, the whole is translated to the right
	{
		int temp = rightBottomx - srcImage.cols + 1;
		rightBottomx -= temp;
		leftTopx -= temp;
	}

	if (rightBottomy > srcImage.rows - 1)//If the lower boundary is exceeded, the whole is translated upward
	{
		int temp = rightBottomy - srcImage.rows + 1;
		rightBottomy -= temp;
		leftTopy -= temp;
	}

	//Define the size of the segmented image, excluding the right corner points
	cv::Mat tempImage(srcImage, Rect(Point(leftTopx, leftTopy), Point(rightBottomx + 1, rightBottomy + 1)));

	cropImage = tempImage.clone();

	return Point(leftTopx, leftTopy);
}

//4. Calculate partial terms of NCC's numerator, where (x, y) are the coordinates of the subgraph points, startRow is the starting row, endRow is the ending row + 1.
double NCCPartTerm(int x, int y, int startRow, int endRow, const Mat & srcImage, const Mat & templateImage)
{
	double NCCVaule = 0;

	Mat srcPartImage = srcImage(Range(y + startRow, y + endRow), Range(x, x + templateImage.cols));
	Mat templatePartImage = templateImage.rowRange(startRow, endRow);

	NCCVaule = srcPartImage.dot(templatePartImage);

	return NCCVaule;
}


//5. Calculate the corresponding value using the box filtering technique,.
cv::Mat boxfilterSqSum(const Mat & srcImage, const Mat & templateImage, int startRow, int endRow)
{
	//prepare work
	Mat srcSqImage;
	srcImage.convertTo(srcSqImage, CV_64FC1);
	cv::pow(srcSqImage, 2, srcSqImage);

	int srcRows = srcImage.rows;
	int srcCols = srcImage.cols;
	int templateRows = templateImage.rows;
	int templateCols = templateImage.cols;

	cv::Mat resultImage;
	int resultRows = srcRows - templateRows + 1;
	int resultCols = srcCols - templateCols + 1;
	resultImage.create(resultRows, resultCols, CV_64FC1);

	//Start use box-filter technique to compute the corresponding vaule
	double *ColBuffer = new double[srcCols];

	for (int i = 0; i < srcCols; i++)
	{
		ColBuffer[i] = 0;
	}

	for (int i = 0; i < srcCols; i++)
	{
		for (int j = startRow; j < endRow; j++)
		{
			ColBuffer[i] += srcSqImage.at<double>(j, i);
		}
	}

	for (int i = 0; i < resultRows; i++)
	{
		resultImage.at<double>(i, 0) = 0;
		for (int k = 0; k < templateCols; k++)
		{
			resultImage.at<double>(i, 0) += ColBuffer[k];
		}

		for (int j = 1; j < resultCols; j++)
		{
			resultImage.at<double>(i, j) = resultImage.at<double>(i, j - 1) - ColBuffer[j - 1] + ColBuffer[templateCols - 1 + j];
		}

		if (i < resultRows - 1)
		{
			for (int m = 0; m < srcCols; m++)
			{
				ColBuffer[m] = ColBuffer[m] - srcSqImage.at<double>(startRow + i, m) + srcSqImage.at<double>(endRow + i, m);//差一点出错
			}
		}
	}

	delete[] ColBuffer;

	return resultImage;
}


//6. Calculate [startRow, endRow) row template pixel sum
double templateSqSum(const Mat & templateImage, int startRow, int endRow)
{
	double sqSum = 0;

	Mat temp = templateImage.rowRange(startRow, endRow);
	sqSum = temp.dot(temp);

	return sqSum;
}

//7. The match processing of BPCcs
cv::Point BPCMatchProcessing(const Mat & srcImage, const Mat & templateImage, double NCCMAX ,double ratio1)
{

	//Determine the boundary of the partition
	int endRow = (int)(templateImage.rows * ratio1);

	
	//Calculate the sum of squares of template pixels and their square roots, and the sum of partial template pixels;
	double partTSum = templateSqSum(templateImage, endRow, templateImage.rows);
	double sqSum = templateSqSum(templateImage,0, templateImage.rows);
	double den_TSum = std::sqrt(sqSum);

	//Calculate subSqSumImage1 and subSqSumImage2 matrices
	Mat subSqSumImage1 = boxfilterSqSum( srcImage, templateImage, 0, endRow);
	Mat subSqSumImage2 = boxfilterSqSum(srcImage, templateImage, endRow, templateImage.rows);


	//Start the template matching process
	Point bestPoint;
	int eliminationPointNums = 0;
	for (int i = 0; i < subSqSumImage1.rows; i++)
	{
		for (int j = 0; j < subSqSumImage1.cols; j++)
		{
			double partTerm = NCCPartTerm(j, i, 0, endRow, srcImage, templateImage);

			double boundTerm2 = cv::sqrt(subSqSumImage2.at<double>(i, j)) * cv::sqrt(partTSum);

			double den_subSum = std::sqrt(subSqSumImage1.at<double>(i, j) + subSqSumImage2.at<double>(i, j));
			double den = den_TSum * den_subSum;

			double upperBoundVaule2 = (partTerm + boundTerm2) / den;
			if (upperBoundVaule2 < NCCMAX)
			{
				eliminationPointNums++;
			}
			else
			{
				double residualTerm = NCCPartTerm(j, i, endRow, templateImage.rows, srcImage, templateImage);//耗时间项
				double NCC = (partTerm + residualTerm) / den;
				if (NCC > NCCMAX)
				{
					NCCMAX = NCC;
					bestPoint = Point(j, i);
				}
			}

		}
	}


	//Calculate the percentage of points eliminated by upper boundary conditions
	double eliminationProportion1 = eliminationPointNums / (double)(subSqSumImage1.rows * subSqSumImage1.cols);

	double speedup1 = (((1 - eliminationProportion1) + eliminationProportion1 * ratio1) * templateImage.rows * templateImage.cols + 8) / (templateImage.rows * templateImage.cols + 4);
	
	return bestPoint;
}



//8. The process of template matching using the search strategy of coarse-fine template matching
//The coarse-fine match process of BPCcs
cv::Point twoStageMatchProcessing(const Mat & srcImage, const Mat & templateImage, double NCCThreshold, double ratio1, int scale)
{
	//Calculate the best sampling factor
	double bestFactor = sampleFactor(srcImage, templateImage);

	
	//Sampling images
	cv::Mat srcSampleImage = subSample(srcImage, bestFactor);
	cv::Mat templateSampleImage = subSample(templateImage, bestFactor);

	//First coarse matching
	cv::Point coarsePoint = BPCMatchProcessing(srcSampleImage, templateSampleImage, NCCThreshold ,ratio1);

	//Segment the source image area for the field of coarsePoint 4 * bestFactor X 4 * bestFactor
	cv::Mat srcNeighborImage;
	cv::Point srcLeftTop = crop_Image(srcImage, templateImage, srcNeighborImage, coarsePoint.x, coarsePoint.y, bestFactor, scale);

	//Then fine matching
	cv::Point finePoint = BPCMatchProcessing(srcNeighborImage, templateImage,NCCThreshold, ratio1);

	//Determine the best match position
	cv::Point matchPoint = srcLeftTop + finePoint;
	cout << "the best match point is " << matchPoint << endl;

	return matchPoint;
}


//main function
int main()
{
	int nums = 32;
	double aver = 0;
	
	for (int i = 0; i < nums; i++)
	{
		cout << "the  " << i + 1 << "  test：" << endl;

		//Read srcImage and templateImage
		Mat srcImage = imread("5-12.bmp", 0);
		Mat templateImage = imread("T5.bmp", 0);

		//Set initialThreshold and correlRation
		double NCCMAX = 0.93;
		double correlRatio = 0.4;
         int scale = 1;

		 //Test matching time
		double start_time = static_cast<double>(getTickCount());
		cv::Point bestPoint = twoStageMatchProcessing(srcImage, templateImage, NCCMAX, correlRatio,scale);

		double consumingTime = 1000 * ((double)getTickCount() - start_time) / getTickFrequency();
		
		cout << "the entire process consumes time: " << consumingTime << "ms" << endl << endl << endl;
	
		if (i > 1)
			aver += consumingTime;
	}

	cout << "The average time: " << aver/(nums-2) << "ms"<<endl;
	cv::waitKey();
	cin.get();
	return 0;
}


