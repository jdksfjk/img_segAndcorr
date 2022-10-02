#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>
#include<opencv2/xfeatures2d.hpp>
#include <string>
#include <iostream>
#include <fstream>
float sigmaG = 4.0, EPSILON = 0.01;

using namespace std;
using namespace cv;
using namespace cv::videostab;
using namespace cv::xfeatures2d;

string inputPath = "E:/imagecorrect/203817/2038/2038.mp4";
string outputPath = "E:/imagecorrect/203817/2038/2038_stable.mp4";
double x_pixel = 0;   //x���������ƫ��
double y_pixel = 0;   //y���������ƫ�� 
static const int MAX_CORNERS = 30;  //�ǵ�ĸ���
double x = 0;
double y = 0;

void SubpixelTranslate(Mat src, Mat& dst, double dx, double dy)
{
	/*
	*src:inputarray
	*dst:outputarray
	* dx:distance in x direction
	* dy:distance in y direction
	*/

	int Rows = src.rows;
	int Cols = src.cols;

	for (int i = 0; i < Rows; i++)
	{
		
		int y = i - dy;//y������ƽ�Ƶ���������
		double u = i - dy - y;//y������ƽ�Ƶ�С������
		if (y >= 0 && y < Rows - 1)  //ƽ�ƺ���ͼ���ڵĲ��ֽ������Բ�ֵ
		{
			uchar* srcdata = src.ptr<uchar>(y);   //ָ���y��
			uchar* srcdata1 = src.ptr<uchar>(y + 1); //ָ���y+1��
			uchar* dstdata = dst.ptr<uchar>(i);
			for (int j = 0; j < Cols - 1; j++)
			{
				int x = j - dx; //x������ƽ�Ƶ���������
				double v = j - dx - x;//x������ƽ�Ƶ�С������
				if (x >= 0 && x < Cols - 1)
				{
					dstdata[j] = (1 - u) * (1 - v) * srcdata[x] + (1 - u) * v * srcdata[x + 1] + u * (1 - v) * srcdata1[x] +
						u * v * srcdata1[x + 1];
				}
			}

		}

	}
}

Mat ImMove(Mat srcImage, int x, int y)
{
	Mat m = Mat::zeros(srcImage.size(), CV_8UC1);
	if (x >= 0 && y >= 0)
	{
		Mat temp = srcImage(Rect(0, 0, srcImage.cols - x, srcImage.rows - y));
		temp.copyTo(Mat(m, Rect(x, y, temp.cols, temp.rows)));
	}
	if (x <= 0 && y >= 0)
	{
		Mat temp = srcImage(Rect(-x, 0, srcImage.cols + x, srcImage.rows - y));
		temp.copyTo(Mat(m, Rect(0, y, temp.cols, temp.rows)));

	}
	if (x >= 0 && y <= 0)
	{
		Mat temp = srcImage(Rect(0, -y, srcImage.cols - x, srcImage.rows + y));
		temp.copyTo(Mat(m, Rect(x, 0, temp.cols, temp.rows)));
	}
	if (x <= 0 && y <= 0)
	{
		Mat temp = srcImage(Rect(-x, -y, srcImage.cols + x, srcImage.rows + y));
		temp.copyTo(Mat(m, Rect(0, 0, temp.cols, temp.rows)));
	}
	return m;
}

void MyFilledCircle(Mat img, Point center)
{
	int thickness = -1;
	int lineType = 8;

	circle(img, center, 512 / 8.0, Scalar(96, 96, 96), thickness, lineType);
}

//video2image
/*
String s1:��Ƶ·��
String s2:��ֺ󱣴���·��
*/
void video2image(String s1, String s2)
{
	VideoCapture capture(s1);
	if (!capture.isOpened())
	{
		std::cout << "open failed!!!" << endl;
	}
	//��ȡ����֡��
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	//��ȡ֡��
	double rate = capture.get(CV_CAP_PROP_FPS);
	//���ÿ�ʼ֡()  
	int frameToStart = 1;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	//���ý���֡
	long frameToStop = totalFrameNumber;
	bool stop = false;
	Mat frame;
	int currentFrame = frameToStart;
	while (!stop)
	{
		//��ȡ��һ֡  
		if (!capture.read(frame))
		{
			std::cout << "��ȡ��Ƶʧ��" << endl;
			return;
		}
		std::cout << "����д��" << currentFrame << "֡" << endl;
		stringstream str;
		str << s2 << currentFrame << ".png";
		imwrite(str.str(), frame);
		if (currentFrame > frameToStop)
		{
			stop = true;
		}
		currentFrame++;
	}
}


int main(int argc, char* argv[])
{
	double time0 = static_cast<double>(getTickCount());
	Mat startImage = imread("E:\\imagecorrect\\practice6_contour\\1_pre.jpg", CV_LOAD_IMAGE_GRAYSCALE);   //��һ֡
	Mat backgroundImage = imread("E:\\imagecorrect\\practice6_contour\\1_pre.jpg"); //��ͼ����
	char image_name[200];
	char result_name[200];
	int win_size = 10;
	Size image_sz = backgroundImage.size();
	Ptr<SURF>detector = SURF::create(3000);
	vector<KeyPoint>keypoints;
	detector->detect(startImage, keypoints);
	vector<Point2f>featureA;
	vector<Point2f>featureB;
	vector<uchar>features_found;
	for (int i = 0; i < keypoints.size(); i++)
	{
		featureA.push_back(keypoints[i].pt);
	}
	for (int j = 0; j < (int)featureA.size(); j++)
	{
		circle(backgroundImage, featureA[j], 5, Scalar(0, 255, 0), 2);
	}
	imshow("result", backgroundImage);
	imwrite("C:\\Users\\ASUS\\Desktop\\cornerdetect2.png", backgroundImage);
	
	for (int i = 2; i <= 298; i++)
	{
		sprintf_s(image_name, "E:\\imagecorrect\\practice4_contour\\(%d)_pre.jpg", i);
		Mat src = imread(image_name, CV_LOAD_IMAGE_GRAYSCALE);
		Mat srcImage = imread(image_name);
		if (src.empty())
		{
			cout << "END����" << endl;
			break;
		}

		//��������
		calcOpticalFlowPyrLK(startImage, src, featureA, featureB, features_found, noArray(), Size(win_size * 2 + 1, win_size * 2 + 1), 5, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.3));                          //��ʱcornersB�д����һ֡�Ľǵ�
		int total = 0;
		for (int i = 0; i < (int)featureA.size(); i++)
		{
			if (!features_found[i])
				continue;
			x_pixel += featureA[i].x - featureB[i].x;
			y_pixel += featureA[i].y - featureB[i].y;
			total++;

		}
		y += y_pixel / total;
		x += x_pixel / total;

		Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
		SubpixelTranslate(src, dst, x, y);
		sprintf_s(result_name, "C:\\Users\\ASUS\\Desktop\\paper_data\\experiment\\srcImage\\results\\drift(%d).tif", i);
		imwrite(result_name, dst);
		startImage = src;
		featureA = featureB;
		featureB.clear();
		features_found.clear();
		x_pixel = 0;
		y_pixel = 0;
	}
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << "Program running time is :" << time0 << " second" << endl;
	imshow("result", backgroundImage);
	imwrite("E:\\imagecorrect\\practice4_contour\\aaa.jpg", backgroundImage);
	waitKey(0)
}





 



