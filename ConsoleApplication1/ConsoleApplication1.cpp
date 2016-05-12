// ConsoleApplication1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;
using namespace std;

int main()
{
	char f1[100], f2[100];
	cin >> f1 >> f2;
	strcat_s(f1, ".jpg");
	strcat_s(f2, ".jpg");
	Mat image1 = imread(f1);
	Mat image2 = imread(f2);

	// 检测surf特征点
	vector<KeyPoint> keypoints1, keypoints2;
	SurfFeatureDetector detector(400);
	detector.detect(image1, keypoints1);
	detector.detect(image2, keypoints2);

	// 描述surf特征点
	SurfDescriptorExtractor surfDesc;
	Mat descriptros1, descriptros2;
	surfDesc.compute(image1, keypoints1, descriptros1);
	surfDesc.compute(image2, keypoints2, descriptros2);

	// 计算匹配点数
	BruteForceMatcher<L2<float>>matcher;
	vector<DMatch> matches;
	matcher.match(descriptros1, descriptros2, matches);
	std::nth_element(matches.begin(), matches.begin() + 24, matches.end());
	matches.erase(matches.begin() + 25, matches.end());

	// 画出匹配图
	Mat imageMatches;
	drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches, Scalar(255, 0, 0));

	for (int i = 0; i < matches.size(); i++) {
		cout << "(" << keypoints1[matches[i].queryIdx].pt.x << "," << keypoints1[matches[i].queryIdx].pt.y << ") <---> ("
			<< keypoints2[matches[i].trainIdx].pt.x << "," << keypoints2[matches[i].trainIdx].pt.y << ")" << endl;
	}

	namedWindow("link_window");
	imshow("link_window", imageMatches);
	waitKey(4000);
	destroyWindow("link_window");

	return 0;
}
