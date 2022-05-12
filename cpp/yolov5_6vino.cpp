#include "yolov5_6vino.h"
 
YOLOVINO::YOLOVINO()
{
    init();
}
 
YOLOVINO::~YOLOVINO()
{
}
 
 
void YOLOVINO::init()
{
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork(m_modelFilename);
    InputsDataMap inputs = network.getInputsInfo();
    OutputsDataMap outputs = network.getOutputsInfo();
    for (auto item : inputs)
    {
        m_inputName = item.first;
        auto input_data = item.second;
        input_data->setPrecision(Precision::FP32);
        input_data->setLayout(Layout::NCHW);
        input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
        //std::cout << "input name = " << m_inputName << std::endl;
    }
 
 
    for (auto item : outputs)
    {
        auto output_data = item.second;
        output_data->setPrecision(Precision::FP32);
        m_outputName = item.first;
        std::cout << "output name = " << m_outputName << std::endl;
    }
    auto executable_network = ie.LoadNetwork(network, m_deviceName);
    m_inferRequest = executable_network.CreateInferRequest();
 
    m_inputData = m_inferRequest.GetBlob(m_inputName);
    m_numChannels = m_inputData->getTensorDesc().getDims()[1];
    m_inputH = m_inputData->getTensorDesc().getDims()[2];
    m_inputW = m_inputData->getTensorDesc().getDims()[3];
    m_imageSize = m_inputH * m_inputW;
	
	loadClassList();
}
 
void YOLOVINO::loadClassList()
{
    std::ifstream ifs(m_classfile);
    std::string line;
    while (getline(ifs, line))
        m_classNames.push_back(line);
}

Mat YOLOVINO::formatYolov5(const Mat &source)
{
	int col = source.cols;
	int row = source.rows;
	int max = MAX(col, row);
    Mat result = Mat::zeros(max, max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
	return result;
}
 
 
void YOLOVINO::detect(Mat &image, vector<Detection> &outputs)
{
    cv::Mat input_image = formatYolov5(image);
    cv::Mat blob_image;
    cv::resize(input_image, blob_image, cv::Size(m_inputW, m_inputH));
    cvtColor(blob_image, blob_image, COLOR_BGR2RGB);
 
    float* data = static_cast<float*>(m_inputData->buffer());
    for (size_t row = 0; row < m_inputH; row++) {
        for (size_t col = 0; col < m_inputW; col++) {
            for (size_t ch = 0; ch < m_numChannels; ch++) {
#ifdef NCS2
				data[m_imageSize * ch + row * m_inputW + col] = float(blob_image.at<cv::Vec3b>(row, col)[ch]);
#else
				data[m_imageSize * ch + row * m_inputW + col] = float(blob_image.at<cv::Vec3b>(row, col)[ch] / 255.0);
#endif // NCS2
            }
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
 
    m_inferRequest.Infer();
    auto output = m_inferRequest.GetBlob(m_outputName);
	const float* detection_out = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
 
    //维度信息
    const SizeVector outputDims = output->getTensorDesc().getDims();//1,6300[25200],9
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "time = " << time << endl;
    float x_factor = float(input_image.cols / m_inputW);
    float y_factor = float(input_image.rows / m_inputH);
    float *dataout = (float *)detection_out;
    const int dimensions = outputDims[2];
    const int rows = outputDims[1];
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    for (int i = 0; i < rows; ++i)
    {
        float confidence = dataout[4];
        if (confidence >= m_confThreshold)
        {
            float * classes_scores = dataout + 5;
            Mat scores(1, m_classNames.size(), CV_32FC1, classes_scores);
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > m_scoreThreshold)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
 
                float x = dataout[0];
                float y = dataout[1];
                float w = dataout[2];
                float h = dataout[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        dataout += dimensions;
    }
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, m_scoreThreshold, m_nmsThreshold, nms_result);
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
		outputs.push_back(result);
    }
}
 
void YOLOVINO::drawRect(Mat &image,vector<Detection> &outputs)
{
    int detections = outputs.size();
    for (int i = 0; i < detections; ++i)
    {
        auto detection = outputs[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        const auto color = colors[classId % colors.size()];
        rectangle(image, box, color, 3);
 
        rectangle(image, Point(box.x, box.y - 40), Point(box.x + box.width, box.y), color, FILLED);
        putText(image, m_classNames[classId].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 0), 2);
    }
 
}