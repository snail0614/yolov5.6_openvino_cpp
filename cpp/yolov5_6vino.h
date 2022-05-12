#ifndef YOLOV5VINO_H
#define YOLOV5VINO_H
#include <fstream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#define NOT_NCS2
using namespace cv;
using namespace dnn;
using namespace std;
using namespace InferenceEngine;
 
class YOLOVINO
{
public:
    struct Detection
    {
        int class_id;
        float confidence;
        Rect box;
    };
public:
    YOLOVINO();
    ~YOLOVINO();
    void init();
    void loadNet(bool is_cuda);
    Mat formatYolov5(const Mat &source);
    void detect(Mat &image,vector<Detection> &outputs);
    void drawRect(Mat &image,vector<Detection> &outputs);
	void loadClassList();
private:
    float m_scoreThreshold = 0.6;
    float m_nmsThreshold = 0.6;
    float m_confThreshold = 0.8;
	
	//"CPU","GPU","MYRIAD"
#ifdef NCS2
	const std::string m_deviceName = "MYRIAD";
	const std::string m_modelFilename = "configFiles/yolov5sNCS2.xml";
#else
	const std::string m_deviceName = "CPU";
	const std::string m_modelFilename = "configFiles/yolov5s.onnx";
#endif // NCS2
	const std::string m_classfile = "configFiles/classes.txt";
    size_t m_numChannels = 0;
    size_t m_inputH = 0;
    size_t m_inputW = 0;
    size_t m_imageSize = 0;
    std::string m_inputName = "";
    std::string m_outputName = "";
    vector<std::string> m_classNames;
    const vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0) };
 
    InferRequest m_inferRequest;
    Blob::Ptr m_inputData;
};
#endif