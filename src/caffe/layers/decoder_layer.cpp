#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/decoder_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DecoderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                     const vector<Blob<Dtype>*>& top) {
  label_size = 20;
}

template <typename Dtype>
void DecoderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                  const vector<Blob<Dtype>*>& top) {
  int batch_size = bottom[0]->count(0, 1);
  top[0]->Reshape(batch_size,label_size);  // (b,l,1,1) -> (b,l)
}

template <typename Dtype>
void DecoderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_label = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  //LOG(INFO) << "Count: " << count;
  for (int i = 0; i < count; i++) {
    int label = bottom_label[i];
    for (int j = label_size - 1; j >= 0; j--) {
      top_data[i*label_size + j] = label & 1;
      label >>= 1;
      //LOG(INFO) << "qDebug: label" << label;
      //LOG(INFO) << "qDebug: top_data" << top_data[i*label_size+j];
    }
  }


}

INSTANTIATE_CLASS(DecoderLayer);
REGISTER_LAYER_CLASS(Decoder);

}  // namespace caffe
