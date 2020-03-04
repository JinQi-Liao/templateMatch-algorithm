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
				ColBuffer[m] = ColBuffer[m] - srcSqImage.at<double>(startRow + i, m) + srcSqImage.at<double>(endRow + i, m);//²îÒ»µã³ö´í
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

//7. The match processing of DBPC
cv::Point BPC2MatchProcessing(const Mat & srcImage, const Mat & templateImage, double NCCThreshold, double  ratio1, double  ratio2)
{
	//Determine the boundaries of the partition
	int rowBound1 = (int)(templateImage.rows * ratio1);
	int rowBound2 = (int)(templateImage.rows *ratio2);

	//Divide the template into three regions and calculate the sum of squared pixels of each region
	double templateSqSum1 = templateSqSum(templateImage, 0, rowBound1);
	double templateSqSum2 = templateSqSum(templateImage, rowBound1, rowBound2);
	double templateSqSum3 = templateSqSum(templateImage, rowBound2, templateImage.rows);

	/*The corresponding sub-window area is also divided into 3 areas, 
	and the box-filtering technique is used to calculate the pixel square sum matrix of each area*/
	cv::Mat subSqSumImage1 = boxfilterSqSum(srcImage, templateImage, 0, rowBound1);
	cv::Mat subSqSumImage2 = boxfilterSqSum(srcImage, templateImage, rowBound1, rowBound2);
	cv::Mat subSqSumImage3 = boxfilterSqSum(srcImage, templateImage, rowBound2, templateImage.rows);

	//Calculate the sum of squares of template pixels and further square root.
	double TSqSum = templateSqSum1 + templateSqSum2 + templateSqSum3;
	double NCC_den1 = std::sqrt(TSqSum);

	//Calculate the sum of squares of sub-window pixels and further square root.
	cv::Mat NCC_den2;
	cv::sqrt((subSqSumImage1 + subSqSumImage2 + subSqSumImage3), NCC_den2);


	//Start searching best match point
	cv::Point bestPoint;
	int eliminationNums1 = 0;
	int eliminationNums2 = 0;
	for (int i = 0; i < subSqSumImage1.rows; i++)
	{
		for (int j = 0; j < subSqSumImage1.cols; j++)
		{
			//Calculate and judge boundary 1 first, and prepare for judgment of boundary 2
			double partTerm1 = NCCPartTerm(j, i, 0, rowBound1, srcImage, templateImage);

			double subBound1 = subSqSumImage2.at<double>(i, j) + subSqSumImage3.at<double>(i, j);
			double TBound1 = templateSqSum2 + templateSqSum3;
			double boundTerm1 = std::sqrt(subBound1) * std::sqrt(TBound1);

			//Compute denominator of NCC
			double NCC_den = NCC_den1 * NCC_den2.at<double>(i, j);

			//Judge boundary 1
			double upperBoundVaule1 = (partTerm1 + boundTerm1) / NCC_den;
			if (upperBoundVaule1 < NCCThreshold)
			{
				eliminationNums1++;
			}
			else
			{
				//Judge boundary 2
				partTerm1 += NCCPartTerm(j, i, rowBound1, rowBound2, srcImage, templateImage);
				subBound1 -= subSqSumImage2.at<double>(i, j);
				TBound1 -= templateSqSum2;
				boundTerm1 = std::sqrt(subBound1) * std::sqrt(TBound1);

				upperBoundVaule1 = (partTerm1 + boundTerm1) / NCC_den;

				if (upperBoundVaule1 < NCCThreshold)
				{
					eliminationNums2++;
				}
				else
				{
					//Judge NCC
					double residualTerm = NCCPartTerm(j, i, rowBound2, templateImage.rows, srcImage, templateImage);
					double NCC = (residualTerm + partTerm1) / NCC_den;
					if (NCC > NCCThreshold)
					{
						NCCThreshold = NCC;
						bestPoint = Point(j, i);
					}
				}
			}

		}
	}

	//Calculate the percentage of points eliminated by upper boundary conditions respectively
	double eliminationProportion1 = eliminationNums1 / (double)(subSqSumImage1.rows * subSqSumImage1.cols);
	double eliminationProportion2 = eliminationNums2 / (double)(subSqSumImage1.rows * subSqSumImage1.cols);


	double speedup = (((1 - eliminationProportion1-eliminationProportion2) + eliminationProportion2 * ratio2 +eliminationProportion1* ratio1)* (templateImage.rows * templateImage.cols) + 12) / (templateImage.rows * templateImage.cols + 4);

	return bestPoint;
}



//8. The process of template matching using the search strategy of coarse-fine template matching
//The coare-fine match process of DBPC, that is TDBPC algorithm
cv::Point twoStageMatchProcessing(const Mat & srcImage, const Mat & templateImage,double NCCThreshold, double ratio1, double ratio2, int scale = 1)
{

	//Calculate the best sampling factor
	double bestFactor = sampleFactor(srcImage, templateImage);
	
	//Sampling images
	cv::Mat srcSampleImage = subSample(srcImage, bestFactor);
	cv::Mat templateSampleImage = subSample(templateImage, bestFactor);

	//First coarse matching
	cv::Point coarsePoint = BPC2MatchProcessing(srcSampleImage, templateSampleImage, NCCThreshold,ratio1, ratio2);

	//Segment the source image area for the field of coarsePoint 4 * bestFactor X 4 * bestFactor
	cv::Mat srcNeighborImage;
	cv::Point srcLeftTop = crop_Image(srcImage, templateImage, srcNeighborImage, coarsePoint.x, coarsePoint.y, bestFactor, scale);

	//Then fine matching
	cv::Point finePoint = BPC2MatchProcessing(srcNeighborImage, templateImage, NCCThreshold, ratio1, ratio2);

	//Determine the best match position
	cv::Point matchPoint = srcLeftTop + finePoint;
	cout << "t match point is: " << matchPoint << endl;

	return matchPoint;
}


int main()
{
	double averTime = 100000;
	int nums = 32;
	for (int i = 0; i < nums; i++)
	{
		cout << " The  " << i + 1 << " test£º" << endl;

	
		//Read srcImage and templateImage
		Mat srcImage = imread("1-7.bmp", 0);
		Mat templateImage = imread("T1.bmp", 0);


		//Set initialThreshold and correlRation
		Point bestPoint;
		double ration2 = 0.4;
		double ration1 = 0.2;
		double NCCMAX = 0.868;
		int scale = 1;

		//Test matching time
		double start_time = static_cast<double>(getTickCount());

	     bestPoint = twoStageMatchProcessing(srcImage, templateImage, NCCMAX, ration1, ration2, scale);

		double consumingTime = 1000 * ((double)getTickCount() - start_time) / getTickFrequency();
		
		cout << endl << endl;
		
		if (consumingTime <= averTime)
			averTime = consumingTime;

		cout << bestPoint << endl << endl << endl;
	}

	cout << "The average time: " << averTime / (nums - 2) << endl;

	waitKey();
	return 0;
}
