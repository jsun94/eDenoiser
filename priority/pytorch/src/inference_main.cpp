#include "test.h"
#include "alex.h"
#include "vgg.h"
#include "resnet.h"
#include "de_resnet.h"
#include "densenet.h"
#include "squeeze.h"
#include "mobile.h"
#include "mnasnet.h"
#include "inception.h"
#include "shuffle.h"
#include "efficient.h"
#include "regnet.h"
#include "origin_denoiser.h"
#include "custom_denoiser.h"
#include "de_vgg.h"
#include "de_inception.h"
#include "de_regnet.h"
#include "co_resnet.h"
#include "co_regnet.h"
#include <cuda_profiler_api.h>

#define n_dense 0
#define n_alex 0
#define n_squeeze 0
#define n_mobile 0
#define n_mnasnet 0
#define n_custom_denoiser 0
// #define n_inception 0
#define n_shuffle 0

#define n_co_res 1
#define n_co_reg 1
#define n_resX 1
#define n_wide 1

#define n_threads 4 // inception 병렬실행하기 위한 최소한의 thread 갯수는 4개
#define WARMING 4

// index of flatten or gap
#define DENSE_FLATTEN 1
#define RES_FLATTEN 1   //WideResNet, ResNext
#define ALEX_FLATTEN 5
#define VGG_FLATTEN 5
#define SQUEEZE_FLATTEN 1
#define MOBILE_FLATTEN 1
#define MNAS_GAP 1
#define INCEPTION_FLATTEN 1
#define SHUFFLE_GAP 1
#define EFFICIENT_FLATTEN 1
#define REG_FLATTEN 1

// #define decompose


extern void *predict_vgg(Net *input);
extern void *predict_de_vgg(Net *input);
extern void *predict_resnet(Net *input);
extern void *predict_co_resnet(Net *input);
extern void *predict_de_resnet(Net *input);
extern void *predict_inception(Net *input);
extern void *predict_de_inception(Net *input);
extern void *predict_regnet(Net *input);
extern void *predict_de_regnet(Net *input);
extern void *predict_origin_denoiser(Net *input);
extern void *predict_custom_denoiser(Net *input);
extern void *predict_decomposed_denoiser(Net *input);
extern void *predict_co_regnet(Net *input);

extern void *predict_warm_inception(Net *input);
extern void *predict_warm_origin_denoiser(Net *input);
extern void *predict_warm_custom_denoiser(Net *input);
extern void *predict_warm_decomposed_denoiser(Net *input);
extern void *predict_warm_regnet(Net *input);
extern void *predict_warm_resnet(Net *input);
extern void *predict_warm_co_resnet(Net *input);
extern void *predict_warm_vgg(Net *input);
extern void *predict_warm_de_vgg(Net *input);
extern void *predict_warm_de_resnet(Net *input);
extern void *predict_warm_de_regnet(Net *input);
extern void *predict_warm_de_inception(Net *input);
extern void *predict_warm_co_regnet(Net *input);

/*extern*/
/* original denoiser */
at::Tensor h_out1;
at::Tensor h_out2;
at::Tensor h_out3;
at::Tensor h_out4;
/* original denoiser */

/* 1x1 denosier */
at::Tensor cu_h_out1;
at::Tensor cu_h_out2;
at::Tensor cu_h_out3;
at::Tensor cu_h_out4;
/* 1x1 denosier */

/* decomposed denoiser */
at::Tensor de_h_out1;
at::Tensor de_h_out2;
at::Tensor de_h_out3;
at::Tensor de_h_out4;
/* decomposed denoiser */

int total_iter;
int total_dnn_iter;
/*extern*/


namespace F = torch::nn::functional;
using namespace std;

threadpool thpool;
pthread_cond_t* cond_t;
pthread_mutex_t* mutex_t;
int* cond_i;
// 2차원 배열형태로 사용하기위해 streams 벡터 다음과 같이 변경
std::vector<std::vector <at::cuda::CUDAStream>> streams;


c10::DeviceIndex GPU_NUM=0;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, const char* argv[]) {
  GPU_NUM=atoi(argv[1]);
  c10::cuda::set_device(GPU_NUM);
  torch::Device device = {at::kCUDA,GPU_NUM};

  // std::string filename = argv[2];


  int n_inception=atoi(argv[2]); //애네들 4명은 HIGH고 나머지는 싹다 LOW
  int n_de_inception=atoi(argv[3]);
  int n_vgg=atoi(argv[4]);
  int n_de_vgg=atoi(argv[5]);
  int n_res=atoi(argv[6]);
  int n_de_res=atoi(argv[7]);
  int n_reg=atoi(argv[8]);
  int n_de_reg=atoi(argv[9]);
  int n_origin_denoiser=atoi(argv[10]);
  int n_decomposed_denoiser=atoi(argv[11]);
  total_iter=atoi(argv[12]);	/* for several input */
  total_dnn_iter=atoi(argv[13]); /* for several other dnns input */

  
  int n_all = n_alex + n_vgg + n_de_vgg + n_res + n_co_res + n_de_res + n_dense + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_de_inception + n_shuffle + n_resX + n_reg + n_de_reg + n_co_reg + n_origin_denoiser + n_custom_denoiser + n_decomposed_denoiser;
  int acc_index_n = 0; // 여기서 acc는 accumulate

  static int stream_index_L = 0;
  static int stream_index_H = 0;
  static int branch_index_L = 31;
  static int branch_index_H = 31;
  static int net_priority_L = n_all-(1 + n_inception + n_origin_denoiser + n_decomposed_denoiser + n_custom_denoiser); // LOW index model get HIHG priority
  //static int net_priority_L = (n_all/2)-1;
  static int net_priority_H = n_all-1;

  streams.resize(2); // streams[a][b], a=1이면 HIGH, a=0이면 LOW, b는 index값, HIGH,LOW각각 32개씩

   /* stream 생성 */
  for(int i=0; i<n_streamPerPool; i++){
    streams[1].push_back(at::cuda::getStreamFromPool(true,GPU_NUM)); //high priority stream  (stream priority 값 = -1)
  }
  for(int i=0; i<n_streamPerPool; i++){
    streams[0].push_back(at::cuda::getStreamFromPool(false,GPU_NUM)); //low priority stream  (stream priority 값 = 0)
  }

  #if FIFOQ
  thpool = thpool_init(n_threads);
  #else
  thpool = thpool_init(n_threads, n_all);
  #endif

  torch::jit::script::Module inceptionModule;
  torch::jit::script::Module origin_denoiserModule;
  torch::jit::script::Module custom_denoiserModule;
  torch::jit::script::Module decomposed_denoiserModule;
  torch::jit::script::Module denseModule;
  torch::jit::script::Module resModule;
  torch::jit::script::Module co_resModule;
  torch::jit::script::Module de_resModule;
  torch::jit::script::Module alexModule;
  torch::jit::script::Module vggModule;
  torch::jit::script::Module de_vggModule;
  torch::jit::script::Module wideModule;
  torch::jit::script::Module squeezeModule;
  torch::jit::script::Module mobileModule;
  torch::jit::script::Module mnasModule;
  torch::jit::script::Module shuffleModule;
  torch::jit::script::Module resXModule;
  torch::jit::script::Module regModule;
  torch::jit::script::Module de_regModule;
  torch::jit::script::Module de_inceptionModule;
  torch::jit::script::Module co_regModule;

  try {
	  	origin_denoiserModule = torch::jit::load("/home/nvidia/joo/HGD_project/traced_origin_denoiser_2.pt", device);
	    origin_denoiserModule.to(device);

	  	custom_denoiserModule = torch::jit::load("/home/nvidia/joo/HGD_project/traced_custom_denoiser_2.pt", device);
	    custom_denoiserModule.to(device);

	  	// decomposed_denoiserModule = torch::jit::load("/home/nvidia/joo/HGD_project/traced_tucker_denoiser.pt", device);
	  	decomposed_denoiserModule = torch::jit::load("/home/nvidia/joo/HGD_project/traced_4tucker_denoiser.pt", device);
	    decomposed_denoiserModule.to(device);

    	inceptionModule = torch::jit::load("../inception_model.pt");
      inceptionModule.to(device);

    	de_inceptionModule = torch::jit::load("/home/nvidia/joo/HGD_project/trace_imagenet_decomposed_inception_model_train.pt", device);
      de_inceptionModule.to(device);

      denseModule = torch::jit::load("/home/nvidia/joo/HGD_project/densenet_model.pt");
      denseModule.to(device);

    	resModule = torch::jit::load("/home/nvidia/joo/HGD_project/resnet_model.pt");
      resModule.to(device);

    	co_resModule = torch::jit::load("/home/nvidia/joo/HGD_project/resnet_model.pt");
      co_resModule.to(device);

	  	de_resModule = torch::jit::load("/home/nvidia/joo/HGD_project/trace_imagenet_res152_decomposed_model.pt");
	  de_resModule.to(device);

    	alexModule = torch::jit::load("/home/nvidia/joo/HGD_project/alexnet_model.pt");
      alexModule.to(device);
  
    	vggModule = torch::jit::load("/home/nvidia/joo/HGD_project/vgg_model.pt");
      vggModule.to(device);

	  	de_vggModule = torch::jit::load("/home/nvidia/joo/HGD_project/trace_imagenet_VGG_decomposed_model.pt");
	  de_vggModule.to(device);

    	wideModule = torch::jit::load("/home/nvidia/joo/HGD_project/wideresnet_model.pt");
      wideModule.to(device);
 
    	squeezeModule = torch::jit::load("/home/nvidia/joo/HGD_project/squeeze_model.pt");
      squeezeModule.to(device);

    	mobileModule = torch::jit::load("/home/nvidia/joo/HGD_project/mobilenet_model.pt");
      mobileModule.to(device);

    	mnasModule = torch::jit::load("/home/nvidia/joo/HGD_project/mnasnet_model.pt");
      mnasModule.to(device);

    	shuffleModule = torch::jit::load("/home/nvidia/joo/HGD_project/shuffle_model.pt");
      shuffleModule.to(device);

    	resXModule = torch::jit::load("/home/nvidia/joo/HGD_project/resnext_model.pt");
      resXModule.to(device);

	  	regModule = torch::jit::load("/home/nvidia/joo/HGD_project/regnet_y_32gf_model.pt");
	  regModule.to(device);

	  	co_regModule = torch::jit::load("/home/nvidia/joo/HGD_project/regnet_y_32gf_model.pt");
	  co_regModule.to(device);

	    de_regModule = torch::jit::load("/home/nvidia/joo/HGD_project/trace_imagenet_regnet_y_32gf_decomposed_model.pt");
	  de_regModule.to(device);
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return -1;
  }
  cout<<"***** Model Load compelete *****"<<"\n";



  cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
  mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
  cond_i = (int *)malloc(sizeof(int) * n_all);


  for (int i = 0; i < n_all; i++)
  {
      pthread_cond_init(&cond_t[i], NULL);
      pthread_mutex_init(&mutex_t[i], NULL);
      cond_i[i] = 0;
  }


  vector<torch::jit::IValue> inputs;
  vector<torch::jit::IValue> inputs2;
  vector<torch::jit::IValue> inputs3;

  torch::Tensor x = torch::ones({1, 3, 224, 224}).to(device);
  inputs.push_back(x);

  torch::Tensor x3 = torch::ones({1, 3, 300, 300}).to(device);
  inputs3.push_back(x3);

  at::Tensor out;

  if(n_de_inception || n_inception || n_origin_denoiser || n_custom_denoiser || n_decomposed_denoiser){
    torch::Tensor x2 = torch::ones({1, 3, 299, 299}).to(device);

    auto x_ch0 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
    auto x_ch1 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
    auto x_ch2 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
      
    x_ch0.to(device);
    x_ch1.to(device);
    x_ch2.to(device);

    auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(device);
    inputs2.push_back(x_cat);
  }
  


  Net net_input_origin_denoiser[n_origin_denoiser];
  Net net_input_custom_denoiser[n_custom_denoiser];
  Net net_input_decomposed_denoiser[n_decomposed_denoiser];
  Net net_input_dense[n_dense];
  Net net_input_res[n_res];
  Net net_input_co_res[n_co_res];
  Net net_input_de_res[n_de_res];
  Net net_input_alex[n_alex];
  Net net_input_vgg[n_vgg];
  Net net_input_de_vgg[n_de_vgg];
  Net net_input_wide[n_wide];
  Net net_input_squeeze[n_squeeze];
  Net net_input_mobile[n_mobile];
  Net net_input_mnasnet[n_mnasnet];
  Net net_input_inception[n_inception];
  Net net_input_de_inception[n_de_inception];
  Net net_input_shuffle[n_shuffle];
  Net net_input_resX[n_resX];
  Net net_input_reg[n_reg];
  Net net_input_de_reg[n_de_reg];
  Net net_input_co_reg[n_co_reg];

  pthread_t networkArray_dense[n_dense];
  pthread_t networkArray_res[n_res];
  pthread_t networkArray_co_res[n_co_res];
  pthread_t networkArray_de_res[n_de_res];
  pthread_t networkArray_alex[n_alex];
  pthread_t networkArray_vgg[n_vgg];
  pthread_t networkArray_de_vgg[n_de_vgg];
  pthread_t networkArray_wide[n_wide];
  pthread_t networkArray_squeeze[n_squeeze];
  pthread_t networkArray_mobile[n_mobile];
  pthread_t networkArray_mnasnet[n_mnasnet];
  pthread_t networkArray_inception[n_inception];
  pthread_t networkArray_de_inception[n_de_inception];
  pthread_t networkArray_shuffle[n_shuffle];
  pthread_t networkArray_resX[n_resX];
  pthread_t networkArray_origin_denoiser[n_origin_denoiser];
  pthread_t networkArray_custom_denoiser[n_custom_denoiser];
  pthread_t networkArray_decomposed_denoiser[n_decomposed_denoiser];
  pthread_t networkArray_reg[n_reg];
  pthread_t networkArray_de_reg[n_de_reg];
  pthread_t networkArray_co_reg[n_co_reg];



  for(int i=0;i<n_dense;i++){
    get_submodule_densenet(denseModule, net_input_dense[i]);
    std::cout << "End get submodule_densenet "<< i << "\n";
    net_input_dense[i].input = inputs;
    net_input_dense[i].name = "DenseNet";
    net_input_dense[i].flatten = net_input_dense[i].layers.size()-1;
    net_input_dense[i].index_n = i+acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_dense[i].H_L = 0; // stream priority의 default값은 low
    net_input_dense[i].index_s = stream_index_L;
    net_input_dense[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
	net_input_dense[i].H_L = 0; 
	net_input_dense[i].index_s = stream_index_L;
	net_input_dense[i].priority = net_priority_L; // net priority는 밑에서부터 올라간다
	stream_index_L+=1;
	net_priority_L-=1;

    
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_densenet(&net_input_dense[i]);
      net_input_dense[i].input = inputs;
    }
    std::cout << "====== END DenseNet WARMUP ======" << std::endl;
  }
  acc_index_n += n_dense;

  for(int i=0;i<n_res;i++){
    get_submodule_resnet(resModule, net_input_res[i]);
    std::cout << "End get submodule_resnet "<< i << "\n";
    net_input_res[i].input = inputs;
    net_input_res[i].name = "ResNet";
    net_input_res[i].flatten = net_input_res[i].layers.size()-1;
    net_input_res[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_res[i].H_L = 0; // stream priority의 default값은 low
    net_input_res[i].index_s = stream_index_L;
    net_input_res[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_res[i].H_L = 1; 
    net_input_res[i].index_s = stream_index_H;
    net_input_res[i].priority = net_priority_H; // net priority는 밑에서부터 올라간다
    stream_index_H+=1;
    net_priority_H-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_resnet(&net_input_res[i]);
      net_input_res[i].input = inputs;
    }
    std::cout << "====== END ResNet WARMUP ======" << std::endl;
  }
  acc_index_n += n_res;

  for(int i=0;i<n_co_res;i++){
    get_submodule_co_resnet(co_resModule, net_input_co_res[i]);
    std::cout << "End get submodule_co_resnet "<< i << "\n";
    net_input_co_res[i].input = inputs;
    net_input_co_res[i].name = "ResNet";
    net_input_co_res[i].flatten = net_input_co_res[i].layers.size()-1;
    net_input_co_res[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_co_res[i].H_L = 0; // stream priority의 default값은 low
    net_input_co_res[i].index_s = stream_index_L;
    net_input_co_res[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_co_res[i].H_L = 0; 
    net_input_co_res[i].index_s = stream_index_L;
    net_input_co_res[i].priority = net_priority_L; // net priority는 밑에서부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_co_resnet(&net_input_co_res[i]);
      net_input_co_res[i].input = inputs;
    }
    std::cout << "====== END CoResNet WARMUP ======" << std::endl;
  }
  acc_index_n += n_co_res;

  for(int i=0;i<n_de_res;i++){
    get_submodule_de_resnet(de_resModule, net_input_de_res[i]);
    std::cout << "End get submodule_de_resnet "<< i << "\n";
    net_input_de_res[i].input = inputs;
    net_input_de_res[i].name = "DeResNet";
    net_input_de_res[i].flatten = net_input_de_res[i].layers.size()-1;
    net_input_de_res[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_de_res[i].H_L = 0; // stream priority의 default값은 low
    net_input_de_res[i].index_s = stream_index_L;
    net_input_de_res[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_de_res[i].H_L = 1; 
    net_input_de_res[i].index_s = stream_index_H;
    net_input_de_res[i].priority = net_priority_H; // net priority는 밑에서부터 올라간다
    stream_index_H+=1;
    net_priority_H-=1;
#endif
    for(int j=0;j<WARMING;j++){
	  std::cout << j << std::endl;
      predict_warm_de_resnet(&net_input_de_res[i]);
      net_input_de_res[i].input = inputs;
    }
    std::cout << "====== END ResNet WARMUP ======" << std::endl;
  }
  acc_index_n += n_de_res;

  for(int i=0;i<n_vgg;i++){
    get_submodule_vgg(vggModule, net_input_vgg[i]);
    std::cout << "End get submodule_vgg "<< i << "\n";
    net_input_vgg[i].input = inputs;
    net_input_vgg[i].name = "VGG";
    net_input_vgg[i].flatten = net_input_vgg[i].layers.size()- VGG_FLATTEN;
    net_input_vgg[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_vgg[i].H_L = 0; // stream priority의 default값은 low
    net_input_vgg[i].index_s = stream_index_L;
    net_input_vgg[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_vgg[i].H_L = 1; 
    net_input_vgg[i].index_s = stream_index_H;
    net_input_vgg[i].priority = net_priority_H; // net priority_L는 0부터 올라간다
    stream_index_H+=1;
    net_priority_H-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_vgg(&net_input_vgg[i]);
      net_input_vgg[i].input = inputs;
    }
    std::cout << "====== END VGG WARMUP ======" << std::endl;
  }
  acc_index_n += n_vgg;

  for(int i=0;i<n_de_vgg;i++){
    get_submodule_de_vgg(de_vggModule, net_input_de_vgg[i]);
    std::cout << "End get submodule_de_vgg "<< i << "\n";
    net_input_de_vgg[i].input = inputs;
    net_input_de_vgg[i].name = "DE_VGG";
    net_input_de_vgg[i].flatten = net_input_de_vgg[i].layers.size()- VGG_FLATTEN;
    net_input_de_vgg[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_de_vgg[i].H_L = 0; // stream priority의 default값은 low
    net_input_de_vgg[i].index_s = stream_index_L;
    net_input_de_vgg[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_de_vgg[i].H_L = 1; 
    net_input_de_vgg[i].index_s = stream_index_H;
    net_input_de_vgg[i].priority = net_priority_H; // net priority_L는 0부터 올라간다
    stream_index_H+=1;
    net_priority_H-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_de_vgg(&net_input_de_vgg[i]);
      net_input_de_vgg[i].input = inputs;
    }
    std::cout << "====== END Decomposed VGG WARMUP ======" << std::endl;
  }
  acc_index_n += n_de_vgg;

  for(int i=0;i<n_alex;i++){
    get_submodule_alexnet(alexModule, net_input_alex[i]);
    std::cout << "End get submodule_alexnet "<< i << "\n";
    net_input_alex[i].input = inputs;
    net_input_alex[i].name = "AlexNet";
    net_input_alex[i].flatten = net_input_alex[i].layers.size()- ALEX_FLATTEN;
    net_input_alex[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_alex[i].H_L = 0; // stream priority의 default값은 low
    net_input_alex[i].index_s = stream_index_L;
    net_input_alex[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_alex[i].H_L = 0; 
    net_input_alex[i].index_s = stream_index_L;
    net_input_alex[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_alexnet(&net_input_alex[i]);
      net_input_alex[i].input = inputs;
    }
    std::cout << "====== END Alex WARMUP ======" << std::endl;
  }
  acc_index_n += n_alex;

  for(int i=0;i<n_squeeze;i++){
    get_submodule_squeeze(squeezeModule, net_input_squeeze[i]);
    std::cout << "End get submodule_squeezenet "<< i << "\n";
    for(int j=0;j<2;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_squeeze[i].record.push_back(event_temp);
    }
    net_input_squeeze[i].input = inputs;
    net_input_squeeze[i].name = "SqueezeNet";
    net_input_squeeze[i].flatten = net_input_squeeze[i].layers.size()- SQUEEZE_FLATTEN;
    net_input_squeeze[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_squeeze[i].H_L = 0; // stream priority의 default값은 low
    net_input_squeeze[i].index_s = stream_index_L;
    net_input_squeeze[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_squeeze[i].H_L = 0; 
    net_input_squeeze[i].index_s = stream_index_L;
    net_input_squeeze[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_squeeze(&net_input_squeeze[i]);
      net_input_squeeze[i].input = inputs;
      for(int n=0;n<net_input_squeeze[i].layers.size();n++){
        net_input_squeeze[i].layers[n].exe_success = false;
      }
    }
    std::cout << "====== END Squeeze WARMUP ======" << std::endl;
  }
  acc_index_n += n_squeeze;

  for(int i=0;i<n_mobile;i++){
    get_submodule_mobilenet(mobileModule, net_input_mobile[i]);
    std::cout << "End get submodule_mobilenet "<< i << "\n";
    net_input_mobile[i].input = inputs;
    net_input_mobile[i].name = "Mobile";
    net_input_mobile[i].flatten = net_input_mobile[i].layers.size()- MOBILE_FLATTEN;
    net_input_mobile[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_mobile[i].H_L = 0; // stream priority의 default값은 low
    net_input_mobile[i].index_s = stream_index_L;
    net_input_mobile[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_mobile[i].H_L = 0; 
    net_input_mobile[i].index_s = stream_index_L;
    net_input_mobile[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_mobilenet(&net_input_mobile[i]);
      net_input_mobile[i].input = inputs;
    }
    std::cout << "====== END Mobile WARMUP ======" << std::endl;
  }
  acc_index_n += n_mobile;

  for(int i=0;i<n_wide;i++){
    get_submodule_resnet(wideModule, net_input_wide[i]);
    std::cout << "End get submodule_widenet "<< i << "\n";
    net_input_wide[i].input = inputs;
    net_input_wide[i].name = "WideResNet";
    net_input_wide[i].flatten = net_input_wide[i].layers.size() - RES_FLATTEN;
    net_input_wide[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_wide[i].H_L = 0; // stream priority의 default값은 low
    net_input_wide[i].index_s = stream_index_L;
    net_input_wide[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_wide[i].H_L = 0; 
    net_input_wide[i].index_s = stream_index_L;
    net_input_wide[i].priority = net_priority_L; // net priority는 밑에서부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_co_resnet(&net_input_wide[i]);
      net_input_wide[i].input = inputs;
    }
    std::cout << "====== END WideRes WARMUP ======" << std::endl;
  }
  acc_index_n += n_wide;

  for(int i=0;i<n_mnasnet;i++){
    get_submodule_MNASNet(mnasModule, net_input_mnasnet[i]);
    std::cout << "End get submodule_mnasnet "<< i << "\n";
    net_input_mnasnet[i].input = inputs;
    net_input_mnasnet[i].name = "MNASNet";
    net_input_mnasnet[i].gap = net_input_mnasnet[i].layers.size() - MNAS_GAP;
    net_input_mnasnet[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_mnasnet[i].H_L = 0; // stream priority의 default값은 low
    net_input_mnasnet[i].index_s = stream_index_L;
    net_input_mnasnet[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_mnasnet[i].H_L = 0; 
    net_input_mnasnet[i].index_s = stream_index_L;
    net_input_mnasnet[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_MNASNet(&net_input_mnasnet[i]);
      net_input_mnasnet[i].input = inputs;
    }
    std::cout << "====== END mnasnet WARMUP ======" << std::endl;
  }
  acc_index_n += n_mnasnet;

  for(int i=0;i<n_resX;i++){
    get_submodule_resnet(resXModule, net_input_resX[i]);
    std::cout << "End get submodule_resXnet "<< i << "\n";
    net_input_resX[i].input = inputs;
    net_input_resX[i].name = "ResNext";
    net_input_resX[i].flatten = net_input_resX[i].layers.size() - RES_FLATTEN;
    net_input_resX[i].index_n = i+acc_index_n;;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_resX[i].H_L = 0; // stream priority의 default값은 low
    net_input_resX[i].index_s = stream_index_L;
    net_input_resX[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_resX[i].H_L = 0; 
    net_input_resX[i].index_s = stream_index_L;
    net_input_resX[i].priority = net_priority_L; // net priority는 밑에서부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_co_resnet(&net_input_resX[i]);
      net_input_resX[i].input = inputs;
    }
    std::cout << "====== END ResNext WARMUP ======" << std::endl;
  }
  acc_index_n += n_resX;

  for(int i=0;i<n_reg;i++){
    get_submodule_regnet(regModule, net_input_reg[i]);
    std::cout << "End get submodule_reg "<< i << "\n";
    net_input_reg[i].input = inputs;
    net_input_reg[i].name = "RegNet";
    net_input_reg[i].flatten = net_input_reg[i].layers.size() - REG_FLATTEN;
    net_input_reg[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_reg[i].H_L = 0; // stream priority의 default값은 low
    net_input_reg[i].index_s = stream_index_L;
    net_input_reg[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_reg[i].H_L = 1; 
    net_input_reg[i].index_s = stream_index_H;
    net_input_reg[i].priority = net_priority_H; // net priority_L는 0부터 올라간다
    stream_index_H+=1;
    net_priority_H-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_regnet(&net_input_reg[i]);
      net_input_reg[i].input = inputs;
    }
    std::cout << "====== END reg WARMUP ======" << std::endl;
  }
  acc_index_n += n_reg;

  for(int i=0;i<n_de_reg;i++){
    get_submodule_de_regnet(de_regModule, net_input_de_reg[i]);
    std::cout << "End get submodule_de_reg "<< i << "\n";
    net_input_de_reg[i].input = inputs;
    net_input_de_reg[i].name = "DeRegNet";
    net_input_de_reg[i].flatten = net_input_de_reg[i].layers.size() - REG_FLATTEN;
    net_input_de_reg[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_de_reg[i].H_L = 0; // stream priority의 default값은 low
    net_input_de_reg[i].index_s = stream_index_L;
    net_input_de_reg[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_de_reg[i].H_L = 1; 
    net_input_de_reg[i].index_s = stream_index_H;
    net_input_de_reg[i].priority = net_priority_H; // net priority_L는 0부터 올라간다
    stream_index_H+=1;
    net_priority_H-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_de_regnet(&net_input_de_reg[i]);
      net_input_de_reg[i].input = inputs;
    }
    std::cout << "====== END de reg WARMUP ======" << std::endl;
  }
  acc_index_n += n_de_reg;

  for(int i=0;i<n_co_reg;i++){
    get_submodule_co_regnet(co_regModule, net_input_co_reg[i]);
    std::cout << "End get submodule_reg "<< i << "\n";
    net_input_co_reg[i].input = inputs;
    net_input_co_reg[i].name = "RegNet";
    net_input_co_reg[i].flatten = net_input_co_reg[i].layers.size() - REG_FLATTEN;
    net_input_co_reg[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_co_reg[i].H_L = 0; // stream priority의 default값은 low
    net_input_co_reg[i].index_s = stream_index_L;
    net_input_co_reg[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_co_reg[i].H_L = 0; 
    net_input_co_reg[i].index_s = stream_index_L;
    net_input_co_reg[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_co_regnet(&net_input_co_reg[i]);
      net_input_co_reg[i].input = inputs;
    }
    std::cout << "====== END Co reg WARMUP ======" << std::endl;
  }
  acc_index_n += n_co_reg;

  for(int i=0;i<n_shuffle;i++){
    get_submodule_shuffle(shuffleModule, net_input_shuffle[i]);
    std::cout << "End get submodule_shuffle "<< i << "\n";
    for(int j=0;j<2;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_shuffle[i].record.push_back(event_temp);
    }
    net_input_shuffle[i].input = inputs;
    net_input_shuffle[i].name = "ShuffleNet";
    net_input_shuffle[i].gap = net_input_shuffle[i].layers.size() - SHUFFLE_GAP;
    net_input_shuffle[i].index_n = i + acc_index_n;
    //priQ는 높은 priority값이 높은 우선순위를 가짐

#if FIFOQ
    // std::cout << "FIFO Q" << std::endl;
    net_input_shuffle[i].H_L = 0; // stream priority의 default값은 low
    net_input_shuffle[i].index_s = stream_index_L;
    net_input_shuffle[i].priority = 0; // FIFO 에서는 priority 설정 X
    stream_index_L +=1;
#else
    net_input_shuffle[i].H_L = 0; 
    net_input_shuffle[i].index_s = stream_index_L;
    net_input_shuffle[i].priority = net_priority_L; // net priority_L는 0부터 올라간다
    stream_index_L+=1;
    net_priority_L-=1;
#endif
    for(int j=0;j<WARMING;j++){
      predict_warm_shuffle(&net_input_shuffle[i]);
      net_input_shuffle[i].input = inputs;
      for(int n=0;n<net_input_shuffle[i].layers.size();n++){
        net_input_shuffle[i].layers[n].exe_success = false;
      }
    }
    std::cout << "====== END shuffle WARMUP ======" << std::endl;
  }
  acc_index_n += n_shuffle;
 
  for(int i=0;i<n_origin_denoiser;i++){
    get_submodule_origin_denoiser(origin_denoiserModule, net_input_origin_denoiser[i]);
    std::cout << "End get submodule_origin_denoiser " << i << "\n";
	  net_input_origin_denoiser[i].input = inputs2;
    net_input_origin_denoiser[i].name = "Original_Denoiser";
    // net_input_vgg[i].flatten = net_input_vgg[i].layers.size() - VGG_FLATTEN;
    net_input_origin_denoiser[i].index_n = i + acc_index_n;
    // net_input_origin_denoiser[i].stream_id = {stream_index_H%n_streamPerPool};
    // stream_index_H+=1;
#if FIFOQ
    net_input_origin_denoiser[i].H_L = 0;
    net_input_origin_denoiser[i].index_s = stream_index_L;
    net_input_origin_denoiser[i].priority = 0; // 무조건 0
    stream_index_L+=1;
#else
    net_input_origin_denoiser[i].H_L = 1; // 무조건 HIGH
    net_input_origin_denoiser[i].index_s = stream_index_H;
    net_input_origin_denoiser[i].priority = net_priority_H;
    stream_index_H+=1;
    net_priority_H-=1;
#endif
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_warm_origin_denoiser(&net_input_origin_denoiser[i]);
      net_input_origin_denoiser[i].input = inputs2;
      
    }

    /*=============FILE===============*/
    //net_input_vgg[i].fp = fopen((filename+"-"+"V"+".txt").c_str(),"a");
  }
  acc_index_n += n_origin_denoiser;

  for(int i=0;i<n_custom_denoiser;i++){
    get_submodule_custom_denoiser(custom_denoiserModule, net_input_custom_denoiser[i]);
    std::cout << "End get submodule_custom_denoiser " << i << "\n";
	  net_input_custom_denoiser[i].input = inputs2;
    net_input_custom_denoiser[i].name = "Custom_Denoiser";
    // net_input_vgg[i].flatten = net_input_vgg[i].layers.size() - VGG_FLATTEN;
    net_input_custom_denoiser[i].index_n = i + acc_index_n;
    // net_input_custom_denoiser[i].stream_id = {stream_index_H%n_streamPerPool};
    // stream_index_H+=1;
#if FIFOQ
    net_input_custom_denoiser[i].H_L = 0;
    net_input_custom_denoiser[i].index_s = stream_index_L;
    net_input_custom_denoiser[i].priority = 0; // 무조건 0
    stream_index_L+=1;
#else
    net_input_custom_denoiser[i].H_L = 1; // 무조건 HIGH
    net_input_custom_denoiser[i].index_s = stream_index_H;
    net_input_custom_denoiser[i].priority = net_priority_H;
    stream_index_H+=1;
    net_priority_H-=1;
#endif
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_warm_custom_denoiser(&net_input_custom_denoiser[i]);
      net_input_custom_denoiser[i].input = inputs2;
      
    }
    acc_index_n += n_custom_denoiser;

    /*=============FILE===============*/
    //net_input_vgg[i].fp = fopen((filename+"-"+"V"+".txt").c_str(),"a");
  }

  

  for(int i=0;i<n_decomposed_denoiser;i++){
    get_submodule_decomposed_denoiser(decomposed_denoiserModule, net_input_decomposed_denoiser[i]);
    std::cout << "End get submodule_decomposed_denoiser " << i << "\n";
	  net_input_decomposed_denoiser[i].input = inputs2;
    net_input_decomposed_denoiser[i].name = "Decomposed_Denoiser";
    // net_input_vgg[i].flatten = net_input_vgg[i].layers.size() - VGG_FLATTEN;
    net_input_decomposed_denoiser[i].index_n = i + acc_index_n;
    // net_input_decomposed_denoiser[i].stream_id = {stream_index_H%n_streamPerPool};
    // stream_index_H+=1;
#if FIFOQ
    net_input_decomposed_denoiser[i].H_L = 0;
    net_input_decomposed_denoiser[i].index_s = stream_index_L;
    net_input_decomposed_denoiser[i].priority = 0; // 무조건 0
    stream_index_L+=1;
#else
    net_input_decomposed_denoiser[i].H_L = 1; // 무조건 HIGH
    net_input_decomposed_denoiser[i].index_s = stream_index_H;
    net_input_decomposed_denoiser[i].priority = net_priority_H;
    stream_index_H+=1;
    net_priority_H-=1;
#endif
    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_warm_decomposed_denoiser(&net_input_decomposed_denoiser[i]);
      net_input_decomposed_denoiser[i].input = inputs2;
      
    }

    /*=============FILE===============*/
    //net_input_vgg[i].fp = fopen((filename+"-"+"V"+".txt").c_str(),"a");
  }
  acc_index_n += n_decomposed_denoiser;

  for(int i=0;i<n_inception;i++){
	  get_submodule_inception(inceptionModule, net_input_inception[i]);
    std::cout << "End get submodule_inception "<< i << "\n";
    for(int j=0;j<4;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_inception[i].record.push_back(event_temp);
    }
    net_input_inception[i].n_all = n_all;
	  net_input_inception[i].input = inputs2;
    net_input_inception[i].name = "Inception_v3";
    net_input_inception[i].flatten = net_input_inception[i].layers.size() - INCEPTION_FLATTEN;
    net_input_inception[i].index_n = i+acc_index_n;;
#if FIFOQ
    net_input_inception[i].H_L = 0; // stream priority의 default값은 low
    net_input_inception[i].index_s = stream_index_L;
    net_input_inception[i].index_b = branch_index_L;
    net_input_inception[i].priority = 0;
    stream_index_L +=1;
    branch_index_L -=3;
#else
    net_input_inception[i].H_L = 1; // HIGH priority stream
    net_input_inception[i].index_s = stream_index_H;
    net_input_inception[i].index_b = branch_index_H;
    net_input_inception[i].priority = net_priority_H;
    stream_index_H+=1;
    branch_index_H-=3;  
    // if(i%2 == 1)
    net_priority_H-=1;
#endif

    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
      predict_warm_inception(&net_input_inception[i]);
      net_input_inception[i].input = inputs2;
      for(int n=0;n<net_input_inception[i].layers.size();n++){
        net_input_inception[i].layers[n].exe_success = false;
      }
      
    }
    /*=============FILE===============*/
    //net_input_inception[i].fp = fopen((filename+"-"+"I"+".txt").c_str(),"a");
  }
  acc_index_n += n_inception;

  for(int i=0;i<n_de_inception;i++){
	  get_submodule_de_inception(de_inceptionModule, net_input_de_inception[i]);
    std::cout << "End get submodule_de_inception "<< i << "\n";
    for(int j=0;j<4;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_de_inception[i].record.push_back(event_temp);
    }
    net_input_de_inception[i].n_all = n_all;
	  net_input_de_inception[i].input = inputs2;
    net_input_de_inception[i].name = "de_Inception_v3";
    net_input_de_inception[i].flatten = net_input_de_inception[i].layers.size() - INCEPTION_FLATTEN;
    net_input_de_inception[i].index_n = i+acc_index_n;;
#if FIFOQ
    net_input_de_inception[i].H_L = 0; // stream priority의 default값은 low
    net_input_de_inception[i].index_s = stream_index_L;
    net_input_de_inception[i].index_b = branch_index_L;
    net_input_de_inception[i].priority = 0;
    stream_index_L +=1;
    branch_index_L -=3;
#else
    net_input_de_inception[i].H_L = 1; // HIGH priority stream
    net_input_de_inception[i].index_s = stream_index_H;
    net_input_de_inception[i].index_b = branch_index_H;
    net_input_de_inception[i].priority = net_priority_H;
    stream_index_H+=1;
    branch_index_H-=3;  
    // if(i%2 == 1)
    net_priority_H-=1;
#endif

    /*=============WARM UP FOR OPTIMIZATION===============*/
    for(int j=0;j<WARMING;j++){
		std::cout << "here" << std::endl;
      predict_warm_de_inception(&net_input_de_inception[i]);
		std::cout << "here2" << std::endl;

      net_input_de_inception[i].input = inputs2;
      for(int n=0;n<net_input_de_inception[i].layers.size();n++){
        net_input_de_inception[i].layers[n].exe_success = false;
      }
      
    }
    /*=============FILE===============*/
    //net_input_inception[i].fp = fopen((filename+"-"+"I"+".txt").c_str(),"a");
  }
  acc_index_n += n_de_inception;

  std::cout<<"\n==================WARM UP END==================\n";
  // cudaDeviceSynchronize();
  
  // cudaProfilerStart();

  cudaEvent_t t_start, t_end;
  float t_time;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start);
  
  //double time1 = what_time_is_it_now();
  for(int i=0;i<n_origin_denoiser;i++){
    if (pthread_create(&networkArray_origin_denoiser[i], NULL, (void *(*)(void*))predict_origin_denoiser, &net_input_origin_denoiser[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_custom_denoiser;i++){
    if (pthread_create(&networkArray_custom_denoiser[i], NULL, (void *(*)(void*))predict_custom_denoiser, &net_input_custom_denoiser[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  
  for(int i=0;i<n_decomposed_denoiser;i++){
    if (pthread_create(&networkArray_decomposed_denoiser[i], NULL, (void *(*)(void*))predict_decomposed_denoiser, &net_input_decomposed_denoiser[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }


  for(int i=0;i<n_dense;i++){
    if (pthread_create(&networkArray_dense[i], NULL, (void *(*)(void*))predict_densenet, &net_input_dense[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_res;i++){
    if (pthread_create(&networkArray_res[i], NULL, (void *(*)(void*))predict_resnet, &net_input_res[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_co_res;i++){
    if (pthread_create(&networkArray_co_res[i], NULL, (void *(*)(void*))predict_co_resnet, &net_input_co_res[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_de_res;i++){
    if (pthread_create(&networkArray_de_res[i], NULL, (void *(*)(void*))predict_de_resnet, &net_input_de_res[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_alex;i++){
    if (pthread_create(&networkArray_alex[i], NULL, (void *(*)(void*))predict_alexnet, &net_input_alex[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_vgg;i++){
	  if (pthread_create(&networkArray_vgg[i], NULL, (void *(*)(void*))predict_vgg, &net_input_vgg[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_de_vgg;i++){
	  if (pthread_create(&networkArray_de_vgg[i], NULL, (void *(*)(void*))predict_de_vgg, &net_input_de_vgg[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_wide;i++){
    if (pthread_create(&networkArray_wide[i], NULL, (void *(*)(void*))predict_co_resnet, &net_input_wide[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_squeeze;i++){
    if (pthread_create(&networkArray_squeeze[i], NULL, (void *(*)(void*))predict_squeeze, &net_input_squeeze[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_mobile;i++){
    if (pthread_create(&networkArray_mobile[i], NULL, (void *(*)(void*))predict_mobilenet, &net_input_mobile[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_mnasnet;i++){
    if (pthread_create(&networkArray_mnasnet[i], NULL, (void *(*)(void*))predict_MNASNet, &net_input_mnasnet[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_inception;i++){
    if (pthread_create(&networkArray_inception[i], NULL, (void *(*)(void*))predict_inception, &net_input_inception[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_de_inception;i++){
    if (pthread_create(&networkArray_de_inception[i], NULL, (void *(*)(void*))predict_de_inception, &net_input_de_inception[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_shuffle;i++){
    if (pthread_create(&networkArray_shuffle[i], NULL, (void *(*)(void*))predict_shuffle, &net_input_shuffle[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_resX;i++){
    if (pthread_create(&networkArray_resX[i], NULL, (void *(*)(void*))predict_co_resnet, &net_input_resX[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_reg;i++){
    if (pthread_create(&networkArray_reg[i], NULL, (void *(*)(void*))predict_regnet, &net_input_reg[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_de_reg;i++){
    if (pthread_create(&networkArray_de_reg[i], NULL, (void *(*)(void*))predict_de_regnet, &net_input_de_reg[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_co_reg;i++){
    if (pthread_create(&networkArray_co_reg[i], NULL, (void *(*)(void*))predict_co_regnet, &net_input_co_reg[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for (int i = 0; i < n_dense; i++){
    pthread_join(networkArray_dense[i], NULL); // pthread_join : thread 종료를 기다리고 thread 종료 이후 다음 진행
  }                                            // join된 thread(종료된 thread)는 모든 resource를 반납

  for (int i = 0; i < n_res; i++){
    pthread_join(networkArray_res[i], NULL);
  }

  for (int i = 0; i < n_co_res; i++){
    pthread_join(networkArray_co_res[i], NULL);
  }

  for (int i = 0; i < n_de_res; i++){
    pthread_join(networkArray_de_res[i], NULL);
  }

  for (int i = 0; i < n_alex; i++){
    pthread_join(networkArray_alex[i], NULL);
  }

  for (int i = 0; i < n_vgg; i++){
    pthread_join(networkArray_vgg[i], NULL);
  }

  for (int i = 0; i < n_de_vgg; i++){
    pthread_join(networkArray_de_vgg[i], NULL);
  }

  for (int i = 0; i < n_wide; i++){
    pthread_join(networkArray_wide[i], NULL);
  }

  for (int i = 0; i < n_squeeze; i++){
    pthread_join(networkArray_squeeze[i], NULL);
  }

  for (int i = 0; i < n_mobile; i++){
    pthread_join(networkArray_mobile[i], NULL);
  }

  for (int i = 0; i < n_mnasnet; i++){
    pthread_join(networkArray_mnasnet[i], NULL);
  }

  for (int i = 0; i < n_inception; i++){
    pthread_join(networkArray_inception[i], NULL);
  }

  for (int i = 0; i < n_de_inception; i++){
    pthread_join(networkArray_de_inception[i], NULL);
  }

  for (int i = 0; i < n_shuffle; i++){
    pthread_join(networkArray_shuffle[i], NULL);
  }

  for (int i = 0; i < n_resX; i++){
    pthread_join(networkArray_resX[i], NULL);
  }
  for (int i = 0; i < n_reg; i++){
    pthread_join(networkArray_reg[i], NULL);
  }
  for (int i = 0; i < n_de_reg; i++){
    pthread_join(networkArray_de_reg[i], NULL);
  }
  for (int i = 0; i < n_co_reg; i++){
    pthread_join(networkArray_co_reg[i], NULL);
  }
  for (int i = 0; i < n_origin_denoiser; i++){
    pthread_join(networkArray_origin_denoiser[i], NULL);
  }

  for (int i = 0; i < n_custom_denoiser; i++){
    pthread_join(networkArray_custom_denoiser[i], NULL);
  }

  for (int i = 0; i < n_decomposed_denoiser; i++){
    pthread_join(networkArray_decomposed_denoiser[i], NULL);
  }

  //cudaDeviceSynchronize();
  //double time2 = what_time_is_it_now();

  cudaDeviceSynchronize();
  cudaEventRecord(t_end);
  cudaEventSynchronize(t_end);
  cudaEventElapsedTime(&t_time, t_start, t_end);

	std::cout << "\n***** TOTAL EXECUTION TIME : "<<t_time/1000<<"s ***** \n";
  // cudaProfilerStop();
  free(cond_t);
  free(mutex_t);
  free(cond_i);

  // for (int i = 0; i < n_dense; i++){
  //   fclose(net_input_dense[i].fp);
  // }
  // for (int i = 0; i < n_res; i++){
  //   fclose(net_input_res[i].fp);
  // }
  // for (int i = 0; i < n_alex; i++){
  //   fclose(net_input_alex[i].fp);
  // }
  // for (int i = 0; i < n_vgg; i++){
  //   fclose(net_input_vgg[i].fp);
  // }
  // for(int i=0;i<n_all;i++){
  //   std::cout<<end_max[i] - start_min[i]<<"\n";
  // }

  // double tmp;
  // double max_value = 0.0;
  
  // for(int i=0;i<n_all;i++){
  //   for(int j=0;j<n_all;j++){
  //     tmp = end_max[i]-start_min[j];
  //     max_value = max(max_value,tmp);
  //   }
  // }
  //  std::cout<<max_value<<"\n";
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include "test.h"
// #include "alex.h"
// #include "vgg.h"
// #include "resnet.h"
// #include "densenet.h"
// #include "squeeze.h"
// #include "mobile.h"
// #include "mnasnet.h"
// #include "inception.h"
// #include "shuffle.h"
// #include "efficient.h"
// #include "regnet.h"
// #include "origin_denoiser.h"
// #include "custom_denoiser.h"
// #include <cuda_profiler_api.h>

// // #define n_dense 1
// // #define n_res 0
// // #define n_alex 0
// // #define n_vgg 0
// // #define n_wide 0
// // #define n_squeeze 0
// // #define n_mobile 0
// // #define n_mnasnet 0
// // #define n_inception 0
// // #define n_shuffle 0
// // #define n_resX 0

// #define n_threads 4
// #define WARMING 4

// // index of flatten or gap
// #define DENSE_FLATTEN 1
// #define RES_FLATTEN 1   //WideResNet, ResNext
// #define ALEX_FLATTEN 5
// #define VGG_FLATTEN 5
// #define SQUEEZE_FLATTEN 1
// #define MOBILE_FLATTEN 1
// #define MNAS_GAP 1
// #define INCEPTION_FLATTEN 1
// #define SHUFFLE_GAP 1
// #define EFFICIENT_FLATTEN 1
// #define REG_FLATTEN 1

// // #define decompose


// extern void *predict_vgg(Net *input);
// extern void *predict_resnet(Net *input);
// extern void *predict_inception(Net *input);
// extern void *predict_regnet(Net *input);
// extern void *predict_origin_denoiser(Net *input);
// extern void *predict_custom_denoiser(Net *input);

// /*extern*/
// at::Tensor h_out1;
// at::Tensor h_out2;
// at::Tensor h_out3;
// at::Tensor h_out4;
// at::Tensor cu_h_out1;
// at::Tensor cu_h_out2;
// at::Tensor cu_h_out3;
// at::Tensor cu_h_out4;
// /*extern*/


// namespace F = torch::nn::functional;
// using namespace std;

// threadpool thpool;
// pthread_cond_t* cond_t;
// pthread_mutex_t* mutex_t;
// int* cond_i;
// std::vector<at::cuda::CUDAStream> streams;

// c10::DeviceIndex GPU_NUM=0;

// double what_time_is_it_now()
// {
//     struct timeval time;
//     if (gettimeofday(&time,NULL)){
//         return 0;
//     }
//     return (double)time.tv_sec + (double)time.tv_usec * .000001;
// }

// int main(int argc, const char* argv[]) {
//   GPU_NUM=atoi(argv[1]);
//   c10::cuda::set_device(GPU_NUM);
//   torch::Device device = {at::kCUDA,GPU_NUM};

//   // std::string filename = argv[2];


//   int n_inception=atoi(argv[2]);
//   int n_origin_denoiser=atoi(argv[3]);
//   int n_custom_denoiser=atoi(argv[4]);

  
//   int n_all = n_inception + n_origin_denoiser + n_custom_denoiser;

//   static int stream_index_H = 0;
//   static int branch_index_H = 31;


//   for(int i=0; i<n_streamPerPool; i++){
//     streams.push_back(at::cuda::getStreamFromPool(true,GPU_NUM));
//   }

//   thpool = thpool_init(n_threads);

//   torch::jit::script::Module inceptionModule;
//   torch::jit::script::Module origin_denoiserModule;
//   torch::jit::script::Module custom_denoiserModule;

//   try {
// 	  	origin_denoiserModule = torch::jit::load("/home/nvidia/joo/HGD_project/traced_origin_denoiser_2.pt", device);
// 	  origin_denoiserModule.to(device);

// 	  	custom_denoiserModule = torch::jit::load("/home/nvidia/joo/HGD_project/traced_custom_denoiser_2.pt", device);
// 	  custom_denoiserModule.to(device);


//     	inceptionModule = torch::jit::load("../inception_model.pt");
//       inceptionModule.to(device);
//   }
//   catch (const c10::Error& e) {
//     cerr << "error loading the model\n";
//     return -1;
//   }
//   cout<<"***** Model Load compelete *****"<<"\n";



//   cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
//   mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
//   cond_i = (int *)malloc(sizeof(int) * n_all);


//   for (int i = 0; i < n_all; i++)
//   {
//       pthread_cond_init(&cond_t[i], NULL);
//       pthread_mutex_init(&mutex_t[i], NULL);
//       cond_i[i] = 0;
//   }


//   vector<torch::jit::IValue> inputs;
//   vector<torch::jit::IValue> inputs2;
//   vector<torch::jit::IValue> inputs3;

//   torch::Tensor x = torch::ones({1, 3, 224, 224}).to(device);
//   inputs.push_back(x);

//   torch::Tensor x3 = torch::ones({1, 3, 300, 300}).to(device);
//   inputs3.push_back(x3);

//   at::Tensor out;

//   if(n_inception || n_origin_denoiser || n_custom_denoiser){
//     torch::Tensor x2 = torch::ones({1, 3, 299, 299}).to(device);

//     auto x_ch0 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
//     auto x_ch1 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
//     auto x_ch2 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
      
//     x_ch0.to(device);
//     x_ch1.to(device);
//     x_ch2.to(device);

//     auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(device);
//     inputs2.push_back(x_cat);
//   }
  

//   Net net_input_inception[n_inception];
//   Net net_input_origin_denoiser[n_origin_denoiser];
//   Net net_input_custom_denoiser[n_custom_denoiser];

//   pthread_t networkArray_inception[n_inception];
//   pthread_t networkArray_origin_denoiser[n_origin_denoiser];
//   pthread_t networkArray_custom_denoiser[n_custom_denoiser];

  
//   for(int i=0;i<n_inception;i++){
// 	  get_submodule_inception(inceptionModule, net_input_inception[i]);
//     std::cout << "End get submodule_inception "<< i << "\n";
//     for(int j=0;j<4;j++){
//       cudaEvent_t event_temp;
//       cudaEventCreate(&event_temp);
//       net_input_inception[i].record.push_back(event_temp);
//     }
//     net_input_inception[i].n_all = n_all;
// 	  net_input_inception[i].input = inputs2;
//     net_input_inception[i].name = "Inception_v3";
//     net_input_inception[i].flatten = net_input_inception[i].layers.size() - INCEPTION_FLATTEN;
//     net_input_inception[i].index_n = i;
//     net_input_inception[i].stream_id = {stream_index_H%n_streamPerPool, abs(branch_index_H)%n_streamPerPool, abs(branch_index_H-1)%n_streamPerPool, abs(branch_index_H-2)%n_streamPerPool};

//     stream_index_H+=1;
//     branch_index_H-=1;

//     /*=============WARM UP FOR OPTIMIZATION===============*/
//     for(int j=0;j<WARMING;j++){
//       predict_inception(&net_input_inception[i]);
//       net_input_inception[i].input = inputs2;
//       for(int n=0;n<net_input_inception[i].layers.size();n++){
//         net_input_inception[i].layers[n].exe_success = false;
//       }
      
//     }
//     /*=============FILE===============*/
//     //net_input_inception[i].fp = fopen((filename+"-"+"I"+".txt").c_str(),"a");
//   }
 
//   for(int i=0;i<n_origin_denoiser;i++){
//     get_submodule_origin_denoiser(origin_denoiserModule, net_input_origin_denoiser[i]);
//     std::cout << "End get submodule_origin_denoiser " << i << "\n";
// 	  net_input_origin_denoiser[i].input = inputs2;
//     net_input_origin_denoiser[i].name = "Original_Denoiser";
//     // net_input_vgg[i].flatten = net_input_vgg[i].layers.size() - VGG_FLATTEN;
//     net_input_origin_denoiser[i].index_n = i + n_inception;
//     net_input_origin_denoiser[i].stream_id = {stream_index_H%n_streamPerPool};
//     stream_index_H+=1;
//     /*=============WARM UP FOR OPTIMIZATION===============*/
//     for(int j=0;j<WARMING;j++){
//       predict_origin_denoiser(&net_input_origin_denoiser[i]);
//       net_input_origin_denoiser[i].input = inputs2;
      
//     }

//     /*=============FILE===============*/
//     //net_input_vgg[i].fp = fopen((filename+"-"+"V"+".txt").c_str(),"a");
//   }

//   for(int i=0;i<n_custom_denoiser;i++){
//     get_submodule_custom_denoiser(custom_denoiserModule, net_input_custom_denoiser[i]);
//     std::cout << "End get submodule_custom_denoiser " << i << "\n";
// 	  net_input_custom_denoiser[i].input = inputs2;
//     net_input_custom_denoiser[i].name = "Custom_Denoiser";
//     // net_input_vgg[i].flatten = net_input_vgg[i].layers.size() - VGG_FLATTEN;
//     net_input_custom_denoiser[i].index_n = i + n_inception + n_origin_denoiser;
//     net_input_custom_denoiser[i].stream_id = {stream_index_H%n_streamPerPool};
//     stream_index_H+=1;
//     /*=============WARM UP FOR OPTIMIZATION===============*/
//     for(int j=0;j<WARMING;j++){
//       predict_custom_denoiser(&net_input_custom_denoiser[i]);
//       net_input_custom_denoiser[i].input = inputs2;
      
//     }

//     /*=============FILE===============*/
//     //net_input_vgg[i].fp = fopen((filename+"-"+"V"+".txt").c_str(),"a");
//   }


//   std::cout<<"\n==================WARM UP END==================\n";
//   cudaDeviceSynchronize();
  
//   cudaProfilerStart();

//   cudaEvent_t t_start, t_end;
//   float t_time;
//   cudaEventCreate(&t_start);
//   cudaEventCreate(&t_end);
//   cudaEventRecord(t_start);
  
//   //double time1 = what_time_is_it_now();
  

//   for(int i=0;i<n_inception;i++){
//     if (pthread_create(&networkArray_inception[i], NULL, (void *(*)(void*))predict_inception, &net_input_inception[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_origin_denoiser;i++){
//     if (pthread_create(&networkArray_origin_denoiser[i], NULL, (void *(*)(void*))predict_origin_denoiser, &net_input_origin_denoiser[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_custom_denoiser;i++){
//     if (pthread_create(&networkArray_custom_denoiser[i], NULL, (void *(*)(void*))predict_custom_denoiser, &net_input_custom_denoiser[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }
  

//   for (int i = 0; i < n_inception; i++){
//     pthread_join(networkArray_inception[i], NULL);
//   }

//   for (int i = 0; i < n_origin_denoiser; i++){
//     pthread_join(networkArray_origin_denoiser[i], NULL);
//   }

//   for (int i = 0; i < n_custom_denoiser; i++){
//     pthread_join(networkArray_custom_denoiser[i], NULL);
//   }

//   //cudaDeviceSynchronize();
//   //double time2 = what_time_is_it_now();

//   cudaDeviceSynchronize();
//   cudaEventRecord(t_end);
//   cudaEventSynchronize(t_end);
//   cudaEventElapsedTime(&t_time, t_start, t_end);

// 	std::cout << "\n***** TOTAL EXECUTION TIME : "<<t_time/1000<<"s ***** \n";
//   cudaProfilerStop();
//   free(cond_t);
//   free(mutex_t);
//   free(cond_i);

//   // for (int i = 0; i < n_dense; i++){
//   //   fclose(net_input_dense[i].fp);
//   // }
//   // for (int i = 0; i < n_res; i++){
//   //   fclose(net_input_res[i].fp);
//   // }
//   // for (int i = 0; i < n_alex; i++){
//   //   fclose(net_input_alex[i].fp);
//   // }
//   // for (int i = 0; i < n_vgg; i++){
//   //   fclose(net_input_vgg[i].fp);
//   // }
//   // for(int i=0;i<n_all;i++){
//   //   std::cout<<end_max[i] - start_min[i]<<"\n";
//   // }

//   // double tmp;
//   // double max_value = 0.0;
  
//   // for(int i=0;i<n_all;i++){
//   //   for(int j=0;j<n_all;j++){
//   //     tmp = end_max[i]-start_min[j];
//   //     max_value = max(max_value,tmp);
//   //   }
//   // }
//   //  std::cout<<max_value<<"\n";
// }
