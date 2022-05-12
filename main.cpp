
#include "inference_engine.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "cpp/yolov5_6vino.h"
using namespace std;
using namespace cv;
using namespace dnn;
using namespace InferenceEngine;

int main(int argc, char** argv)
{
	YOLOVINO yolov5vino;
	int frameCount = 0;
	int fps = 0;
	Mat frame;

	auto start = std::chrono::high_resolution_clock::now();
	VideoCapture capture("sample.mp4");
	while (1)
	{
		capture.read(frame);
		if (frame.empty())
		{
			break;
		}
		++frameCount;
		if (frame.channels() < 3)
			cvtColor(frame, frame, COLOR_GRAY2RGB);

		std::vector<YOLOVINO::Detection> outputs;
		yolov5vino.detect(frame, outputs);
		yolov5vino.drawRect(frame, outputs);
		if (frameCount >= 15)
		{

			auto end = std::chrono::high_resolution_clock::now();
			fps = frameCount * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			frameCount = 0;
			start = std::chrono::high_resolution_clock::now();
		}

		if (fps > 0)
		{

			std::ostringstream fps_label;
			fps_label << std::fixed << std::setprecision(2);
			fps_label << "FPS: " << fps;
			std::string fps_label_str = fps_label.str();
			cv::putText(frame, fps_label_str.c_str(), cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);
		}
		cv::namedWindow("output", cv::WINDOW_NORMAL);
		cv::imshow("output", frame);

		if (cv::waitKey(1) != -1)
		{
			capture.release();
			std::cout << "finished by user\n";
			break;
		}
	}
	return 1;
}

