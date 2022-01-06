#include <iostream>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

int main(){
    vector<float> b = {1.0f, 2.0f,3.0,4.0,5,6,7,8,9,10,11,12,13,14,15,16};
    Mat mat2(4,4, CV_32F,b.data());
    auto f2 = mat2.rowRange(0, 4).colRange(0, 2);
    auto s2 = mat2.rowRange(0, 4).colRange(2, 4);
    cout << f2 << endl;
    cout<<s2 << endl;
    cout << (f2*2).mul(s2) << endl;

}
