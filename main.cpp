#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "detector.h"
#include "priorbox.h"
#include <fstream>
using namespace std;
using namespace cv;
int main()
{
    string model_path = "/Users/mix/FaceBoxes.PyTorch/FaceBoxes.onnx";
    string image_path = "data/test.jpg";
    Mat clr = imread(image_path,IMREAD_COLOR);
    Mat image;
    Mat conf;
    Mat loc;
    resize(clr, image, Size(), 2.5, 2.5);


    ParamConfig config = {0.8f, 0.3f, model_path};
    vector<vector<float>> m_sizes = {{32, 64, 128}, {256}, {512}};
    vector<float> steps = {32.0, 64.0, 128.0};
    vector<float> variance = {0.1, 0.2};
    PriorBox pbox(Size(image.cols, image.rows), m_sizes, steps, false);
    Mat anchors;
    pbox.forward(anchors);

    Detector detector(config);
    detector.detect(image, loc, conf);
    ofstream ofs("out.csv");
    for (int i = 0; i < conf.rows; i++)
    {
        if (conf.at<float>(i, 1) > 0.99)
        {
            ofs << i << "," << loc.at<float>(i, 0) << "," << loc.at<float>(i, 1) << "," << loc.at<float>(i, 2) << "," << loc.at<float>(i, 3) << "," << conf.at<float>(i, 1) << endl;
        }
    }
    ofs.close();
    cout << anchors.row(15349) << endl;
    auto priors = pbox.decode(loc, anchors, variance, image.cols, image.rows);

    cout << priors.row(15349) << endl;
    detector.postProcess(priors, conf, clr);
    imshow("detection", clr);
    waitKey(0);
}
