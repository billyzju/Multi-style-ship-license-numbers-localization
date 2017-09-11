
The codes and the transferred deep model released here are described in our ICTAI 2017 paper "Using transferred deep model in combination with prior features to localize
multi-style ship license numbers in nature scenes", please refer our paper for more details.

1. The Transferred_Deep_Model was implemented and trained with Caffe (TextBoxes version of Caffe, see also https://github.com/MhLiao/TextBoxes).
2. How to use ?
   To combine our transferred deep model and other two  SLNs prior-based SLNs region generating and fake SLNs filtering algorithms (they are implemented with C++),
we use an Eclipse IDE for C/C++ Developers. By calling our transferred deep model and two algorithms seamlessly, we can conduct an end-to-end style testin on this Eclipse platform.

You are advised to figure out how to use C++ to handle Caffe-based projects.You may find a sample code in "~/caffe/examples/ssd/ssd_detect.cpp" from SSD (https://github.com/weiliu89/caffe/tree/ssd).
Once you figure this point out, it will be easy for you to run our codes and transferred deep model with an Eclipse IDE for C/C++.

Please let me know if you encounter any issues.

P.S.: Our transferred deep model is too big to be uploaded on Github, you can find it at http://pan.baidu.com/s/1i4YJHLb
