#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
using namespace std;
using namespace cv;
int main()
{
    string model_path = "/Users/mix/FaceBoxes.PyTorch/FaceBoxes.onnx";
    string image_path = "data/test.jpg";
    Mat clr = imread(image_path,IMREAD_COLOR);
    auto info_txt = getBuildInformation();
    ofstream info("buildinfo.txt");
    info << info_txt;
    info.close();
    Mat image, blob, conf, loc;
    resize(clr, image, Size(), 2.5, 2.5,INTER_LINEAR);
    dnn::blobFromImage(image, blob, 1.0, Size(image.cols, image.rows), Scalar(104.0, 117.0, 123.0), false, false);
    auto net = dnn::readNetFromONNX("FaceBoxes.onnx");
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    net.setInput(blob);
    std::vector<string> outLayerNames = net.getUnconnectedOutLayersNames();
    std::vector<Mat> outs;
    net.forward(outs, outLayerNames);
    Mat(outs[0].size[1], outs[0].size[2], CV_32F, outs[0].data).copyTo(loc);
    Mat(outs[1].size[1], outs[1].size[2], CV_32F, outs[1].data).copyTo(conf);
    ofstream ofs("out.csv");
    for (int i = 0; i < conf.rows; i++)
    {
        if (conf.at<float>(i, 1) > 0.999)
        {
            ofs << i << "," << loc.at<float>(i, 0) << "," << loc.at<float>(i, 1) << "," << loc.at<float>(i, 2) << "," << loc.at<float>(i, 3) << "," << conf.at<float>(i, 1) << endl;
        }
    }
    ofs.close();
}
