#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>

#define SRC_WINDOWNAME		"Original"
#define ENCRYPT_WINDOWNAME	"Encryption"
#define DECRYPT_WINDOWNAME	"Decryption"
#define INPUT_HIST_WINDOWNAME "Input Histogram"
#define OUTPUT_HIST_WINDOWNAME "Output Histogram"

using namespace cv;

int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

clock_t deltaTime = 0;

int imageEncryptionTest();
int videoEncryptionTest();
cv::Mat generateKeyImg(int width, int height);
cv::Mat getImgKeyFromFile(int width, int height);
Mat Encrypt(Mat img, Mat key);
Mat Decrypt(Mat image, Mat key);
Mat displayHistogram(Mat frame);
void saveImg(Mat input, Mat output);

int main()
{
	//uncomment one of the lines below to demo
	imageEncryptionTest();
	//videoEncryptionTest();

	return 0;
}

int imageEncryptionTest() {
	Mat src;
	//load image
	src = imread("lena.png", CV_LOAD_IMAGE_COLOR);
	if (src.empty()) { return -1; }

	//convert image to grayscale
	src.convertTo(src, CV_8U);
	int src_size = (src.rows * src.cols);
	Mat keyImg = generateKeyImg(32, 32);

	//Mat keyImg = getImgKeyFromFile(32,32);

	//resize image to source image size before encrypting
	resize(keyImg, keyImg, Size(src.cols, src.rows));

	Mat src_encrypt = Encrypt(src, keyImg);
	Mat src_decrypt = Decrypt(src_encrypt, keyImg);

	Mat src_hist, encrypted_hist;
	src_hist = displayHistogram(src);
	encrypted_hist = displayHistogram(src_encrypt);

	namedWindow(SRC_WINDOWNAME, WINDOW_AUTOSIZE);
	namedWindow(ENCRYPT_WINDOWNAME, WINDOW_AUTOSIZE);
	namedWindow(DECRYPT_WINDOWNAME, WINDOW_AUTOSIZE);
	namedWindow(INPUT_HIST_WINDOWNAME, CV_WINDOW_AUTOSIZE);
	namedWindow(OUTPUT_HIST_WINDOWNAME, CV_WINDOW_AUTOSIZE);

	//display source image
	imshow(SRC_WINDOWNAME, src);
	//display source histogram
	imshow(INPUT_HIST_WINDOWNAME, src_hist);
	imshow(ENCRYPT_WINDOWNAME, src_encrypt);
	imshow(OUTPUT_HIST_WINDOWNAME, encrypted_hist);
	imshow(DECRYPT_WINDOWNAME, src_decrypt);
	saveImg(src, src_encrypt);

	for (;;)
	{
		char c = (char)waitKey(10);
		if (c == 27)
		{
			break;
		}
	}

	return 0;
}
int videoEncryptionTest() {
	int count = 0;
	Mat imgKey, imgEncrypted;
	VideoWriter output;
	VideoCapture cap("drive.mp4");
	if (!cap.isOpened()) {
		return -1;
	}
	//loop until video ends
	for (;;)
	{
		Mat frame;
		cap >> frame; 
		//check if frame is empty/video is over
		if (frame.empty()) {
			break;
		}
		//check if first loop
		if (count == 0) {
			imgKey = generateKeyImg(frame.cols, frame.rows);
		}

		//encrypt frame
		imgEncrypted = Encrypt(frame, imgKey);

		//write output video
		imshow("out", imgEncrypted);

		char c = (char)waitKey(1);
		if (c == 27)
		{
			break;
		}
		count++;
	}
	return 0;
}


Mat Encrypt(Mat image, Mat key) {

	clock_t beginFrame = clock();
	Mat encrypted = image.clone();
	uchar pixValue;
	int counter = 0;
	for (int i = 0; i < encrypted.cols; i++) {
		for (int j = 0; j < encrypted.rows; j++) {
			Vec3b &intensity_img = encrypted.at<Vec3b>(j, i);
			Vec3b &intensity_key = key.at<Vec3b>(j, i);
			for (int k = 0; k < encrypted.channels(); k++) {
				// calculate pixValue
				intensity_img.val[k] += intensity_key.val[k];
			}
			counter++;
		}
	}

	clock_t endFrame = clock();
	deltaTime = endFrame - beginFrame;
	std::cout << "Encryption time: " << deltaTime << "ms" << std::endl;
	return encrypted;
}



Mat Decrypt(Mat image, Mat key) {

	Mat decrypted = image.clone();
	clock_t beginFrame = clock();
	uchar pixValue;
	int counter = 0;
	for (int i = 0; i < decrypted.cols; i++) {
		for (int j = 0; j < decrypted.rows; j++) {
			Vec3b &intensity_img = decrypted.at<Vec3b>(j, i);
			Vec3b &intensity_key = key.at<Vec3b>(j, i);
			for (int k = 0; k < decrypted.channels(); k++) {
				// calculate pixValue
				intensity_img.val[k] -= intensity_key.val[k];
			}
			counter++;
		}
	}
	clock_t endFrame = clock();
	deltaTime = endFrame - beginFrame;
	std::cout << "Decryption time: " << deltaTime << "ms" << std::endl;
	return decrypted;
}


cv::Mat generateKeyImg(int width, int height) {
	std::ofstream keyFile;
	keyFile.open("key.txt");
	
	Mat output = Mat::zeros(width, height,CV_64F);
	uchar pixValue;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			Vec3b &intensity = output.at<Vec3b>(j, i);
			for (int k = 0; k < 3; k++) {
				// calculate pixValue
				pixValue =  rand() % 255;
				intensity.val[k] = pixValue;
				keyFile << pixValue << std::endl;
				//std::cout << pixValue << std::endl;
			}
		}
	}
	keyFile.close();
	return output;
}

cv::Mat getImgKeyFromFile(int width, int height) {
	std::ifstream keyFile;
	keyFile.open("key.txt");

	Mat output = Mat::zeros(width, height, CV_64F);
	uchar pixValue;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			Vec3b &intensity = output.at<Vec3b>(j, i);
			for (int k = 0; k < 3; k++) {
				// calculate pixValue
				//pixValue = rand() % 255;
				
				keyFile >> pixValue;
				intensity.val[k] = pixValue;
				//std::cout << pixValue << std::endl;
			}
		}
	}
	keyFile.close();
	return output;
}



void saveImg(Mat input, Mat output) {
	cv::imwrite("input.png", input);
	cv::imwrite("output.png", output);
}

Mat displayHistogram(cv::Mat frame) {
	Mat dst;

	// prepare for fourier transform
	std::vector<Mat> bgr_planes;
	split(frame, bgr_planes);

	// Establish the number of bins
	int histSize = 256;

	// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	// calculate histogram
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);


	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	return histImage;

}
