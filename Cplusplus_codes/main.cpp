/*
 * main.cpp
 *
 *  Created on: Jun 7, 2017
 *      Author: cgim
 */

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//eclipse
#include "SLNsLocate.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {

  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(static_cast<int> (values.size()) == 1 || static_cast<int> (values.size()) == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");


int main(int argc, char** argv) {

	//Prepare the deploy files of each model.
	//You may change the path below with YOUR LOCAL PATH!

    string dp300 = "/home/cgim/caffe/models/VGGNet/ships/deploy300.prototxt";
	string dp500 = "/home/cgim/caffe/models/VGGNet/ships/deploy500.prototxt";
	string dp700 = "/home/cgim/caffe/models/VGGNet/ships/deploy700.prototxt";
	string dp1000 = "/home/cgim/caffe/models/VGGNet/ships/deploy1000.prototxt";

  //weights_file. Change the path with YOUR LOCAL PATH!
  const string& weights_file = "/home/cgim/caffe/models/VGGNet/ships/ships_mlt_300x300/ships_mlt_300x300_iter_60000.caffemodel";//argv[2];

  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);


  vector<vector<cv::Rect> >  locatedRects;
  vector<Rect> detPimg;
  PARAM para;

  //super parameters for SLNs prior-based fine location.
  SPPARA sppaTest = {5,5,0.45,0.175,2,13,50,10,0.35,0.35};

  para.id = 1;
  sppaTest.id = 1;

  Rect cur;
  cur.x = 0;
  cur.y = 0;
  cur.width = 0;
  cur.height = 0;
  vector<cv::Rect> emptyImg;
  emptyImg.push_back(cur);

  vector<Detector> adaTest;
  vector<float> confTest600,confTest350;

  //models initialization
  const string& model_file300 = dp300;
  Detector detector300(model_file300, weights_file, mean_file, mean_value);

  const string& model_file500 = dp500;
  Detector detector500(model_file500, weights_file, mean_file, mean_value);


  const string& model_file700 = dp700;
  Detector detector700(model_file700, weights_file, mean_file, mean_value);

  const string& model_file1000 = dp1000;
  Detector detector1000(model_file1000, weights_file, mean_file, mean_value);

  adaTest.push_back(detector300);adaTest.push_back(detector500);
  adaTest.push_back(detector700);adaTest.push_back(detector1000);

  //confidence thresholds of different models.
  confTest600.push_back(0.5);confTest600.push_back(0.3);
  confTest600.push_back(0.3);confTest600.push_back(0.4);
  confTest350.push_back(0.4);confTest350.push_back(0.4);
  confTest350.push_back(0.45);confTest350.push_back(0.9);


  // Process image one by one. Change the path with YOUR LOCAL PATH!
  std::string imagePath = "/home/cgim/caffe/data/ships/shipAll/";
  std::ifstream infile("/home/cgim/caffe/data/ships/shipAll/list.txt"/*argv[3]*/);
  std::string file;

  double t_toal = (double)cvGetTickCount();

  while (infile >> file) {

	  if(para.id < 0)
	  {
		  para.id++;
		  continue;
	  }
	  if(para.id > STOP)
		  break;

	  detPimg.clear();

    if (file_type == "image") {

      cv::Mat img = cv::imread(imagePath+file,-1);

      CHECK(!img.empty()) << "Unable to decode image " << file;

      Mat imgCopy,final;
      img.copyTo(imgCopy);
      img.copyTo(final);

      double t = (double)cvGetTickCount();

      vector<Detector> adaDcur;
      vector<float> confCur;
      SPPARA sppa;
      adaDcur.clear();
      confCur.clear();

      adaDcur = adaTest;
      sppa = sppaTest;

     if(para.id < 601)
      {
    	  confCur = confTest600;
      }
      else
      {
    	  confCur = confTest350;
      }


      for (int f = 0; f < static_cast<int> (adaDcur.size()); f++)
      {

      std::vector<vector<float> > detections = adaDcur[f].Detect(img);

       //Print the detection results.
      for (int i = 0; i < static_cast<int> (detections.size()); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        float score = d[2];

        if (score >= confCur[f]) {
          out << file << " ";
          out << static_cast<int>(d[1]) << " ";
          out << score << " ";
          out << static_cast<int>(d[3] * img.cols) << " ";
          out << static_cast<int>(d[4] * img.rows) << " ";
          out << static_cast<int>(d[5] * img.cols) << " ";
          out << static_cast<int>(d[6] * img.rows) << std::endl;

          int xmin, ymin, xmax, ymax;
          xmin = static_cast<int>(d[3] * img.cols);
          ymin = static_cast<int>(d[4] * img.rows);
          xmax = static_cast<int>(d[5] * img.cols);
          ymax = static_cast<int>(d[6] * img.rows);


          if(xmin < 0)
       	     xmin = 0;
          if(ymin < 0)
       	     ymin =0;
          if(xmax > img.cols)
       	     xmax = img.cols;
          if(ymax > img.rows)
       	     ymax = img.rows;

          Rect thisImg;
          thisImg.x = xmin;
          thisImg.y = ymin;
          thisImg.width = xmax - xmin;
          thisImg.height = ymax - ymin;
          detPimg.push_back(thisImg);

        //  cv::rectangle(imgCopy,cvPoint(static_cast<int>(d[3] * img.cols),static_cast<int>(d[4] * img.rows)),cvPoint(static_cast<int>(d[5] * img.cols),static_cast<int>(d[6] * img.rows)),cv::Scalar(0,0,255),2,1,0);
         }
      }
   }

      para.filename = file;
      SLNsLocate pImg;

      if(detPimg.empty())
      {
          locatedRects.push_back(emptyImg);
      }
      else    //SLNs prior-based fine location
      {
    	  vector<cv::Rect> MergedRects,filterred;
    	  MergedRects = pImg.rectsMerge(img, detPimg, sppa);   //Comprehensive SLNs regions generating
    	  filterred = pImg.SLNsFiltering(img, MergedRects, sppa);  //Fake SLNs filtering

    	  if(filterred.empty())
    		  locatedRects.push_back(emptyImg);
    	  else
    		  locatedRects.push_back(filterred);
      }

      t = (cvGetTickCount() - t)/((double)cvGetTickFrequency()*1000.);
      std::cout << setiosflags(ios::fixed);
      std::cout << "Time of this image: " << std::setprecision(3) << t << std::endl;

    }
	  para.id++;
	  sppaTest.id++;
 }

  t_toal = ((double)cvGetTickCount() - t_toal)/((double)cvGetTickFrequency()*1000.);
  std::cout << "Time of ALL: " << t_toal << std::endl;

  return 0;
}

