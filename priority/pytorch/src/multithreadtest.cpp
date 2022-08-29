
// #include <cuda_runtime.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAMultiStreamGuard.h>
// #include <ATen/cuda/CUDAEvent.h>
// #include <c10/cuda/CUDAStream.h>
// #include <pthread.h>
// #include <torch/script.h>
// #include <torch/torch.h>
// #include <typeinfo>
// #include <iostream>
// #include <inttypes.h>
// #include <functional>
// #include <memory>
// #include <stdlib.h>
// #include <c10/cuda/CUDAFunctions.h>
// #include <cuda_profiler_api.h>
// #include <limits.h>
// #include <time.h>
// #include <sys/time.h>

// #include <cuda_runtime.h>
// #include <thread>

// #include <limits.h>
// #include <time.h>
// #include <sys/time.h>
// #include <algorithm>


// #define n_dense 0
// #define n_res 0
// #define n_alex 0
// #define n_vgg 10
// #define n_wide 0
// #define n_squeeze 0
// #define n_mobile 0
// #define n_mnasnet 0
// #define n_inception 0
// #define n_shuffle 0
// #define n_resX 0

// #define WARMING 0

// namespace F = torch::nn::functional;
// using namespace std;

// c10::DeviceIndex GPU_NUM = 0;

// vector<torch::jit::IValue> inputs;
// vector<torch::jit::IValue> inputs2;
// torch::Device device = {at::kCUDA,GPU_NUM};
// torch::Tensor x = torch::ones({1, 3, 224, 224}).to(device);
// torch::Tensor x2 = torch::ones({1, 3, 299, 299}).to(device);

// typedef struct _net
// {
// 	torch::jit::script::Module module;
// 	std::vector<torch::jit::IValue> input;
// 	std::string name;	//network name
// 	int index_n; //network index
// 	int n_all; // all network num
// }Net;

// double what_time_is_it_now()
// {
//     struct timeval time;
//     if (gettimeofday(&time,NULL)){
//         return 0;
//     }
//     return (double)time.tv_sec + (double)time.tv_usec * .000001;
// }

// const int n_all = n_alex + n_vgg + n_res + n_dense + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle + n_resX;

// double *start_time;
// double *end_time;

// void *predict_inception(Net *net){
//     at::Tensor out;
//     out = net->module.forward(net->input).toTensor();
//     std::cout << out.slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";	
//     //cout<<t2-t1<<"\n";
// }

// std::condition_variable g_controller;

// void *network_predict(Net *net){
//   // std::mutex g_mutex;
//   // std::unique_lock<std::mutex> lock(g_mutex);
//   // cout<<"net->index = "<<net->index_n<<"\n";
//   // g_controller.wait(lock); 

//   std::cout<<net->name<<" = "<<&(net->module)<<"\n";
//   start_time[net->index_n] = what_time_is_it_now();
//   at::Tensor out;
//   cout<<"size = "<<net->input[0].toTensor().sizes()<<"\n";
//   out = net->module.forward(net->input).toTensor();
//   end_time[net->index_n] = what_time_is_it_now();
//   //cout<<end_time[net->index_n]<<"\n";
//   std::cout << out.slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";	
//   std::cout << "\n***** "<<net->name<<" EXECUTION TIME : "<< end_time[net->index_n]-start_time[net->index_n] <<"s ***** \n";
// }



// int main(){    
//     if (torch :: cuda :: is_available ()) {
//       cout << torch :: cuda :: is_available () << endl;
//     }
//     cout<<"start!!!!!!!!!!!!!!!!\n";
//     //c10::cuda::set_device(GPU_NUM);
//     start_time = (double *)malloc(sizeof(double)*n_all);
//     end_time = (double *)malloc(sizeof(double)*n_all);
//     torch::jit::script::Module model[11];
//     model[0] = torch::jit::load("../densenet201_model.pt");
//     model[0].to(device);
//     model[1] = torch::jit::load("../resnet152_model.pt");
//     model[1].to(device);

//     // model[1] = torch::jit::load("../densenet201_model.pt");
//     // model[1].to(device);

//     model[2] = torch::jit::load("../alexnet_model.pt");
//     model[2].to(device);
//     model[3] = torch::jit::load("../vgg_model.pt");
//     model[3].to(device);
//     model[4] = torch::jit::load("../wideresnet_model.pt");
//     model[4].to(device);
//     model[5] = torch::jit::load("../squeeze_model.pt");
//     model[5].to(device);
//     model[6] = torch::jit::load("../mobilenet_model.pt");
//     model[6].to(device);
//     model[7] = torch::jit::load("../mnasnet_model.pt");
//     model[7].to(device);
//     model[8] = torch::jit::load("../inception_model.pt");
//     model[8].to(device);
//     model[9] = torch::jit::load("../shuffle_model.pt");
//     model[9].to(device);
//     model[10] = torch::jit::load("../resnext_model.pt");
//     model[10].to(device);

//     pthread_t networkArray_dense[n_dense];
//     pthread_t networkArray_res[n_res];
//     pthread_t networkArray_alex[n_alex];
//     pthread_t networkArray_vgg[n_vgg];
//     pthread_t networkArray_wide[n_wide];
//     pthread_t networkArray_squeeze[n_squeeze];
//     pthread_t networkArray_mobile[n_mobile];
//     pthread_t networkArray_mnasnet[n_mnasnet];
//     pthread_t networkArray_inception[n_inception];
//     pthread_t networkArray_shuffle[n_shuffle];
//     pthread_t networkArray_resX[n_resX];

//     Net net_input_dense[n_dense];
//     Net net_input_res[n_res];
//     Net net_input_alex[n_alex];
//     Net net_input_vgg[n_vgg];
//     Net net_input_wide[n_wide];
//     Net net_input_squeeze[n_squeeze];
//     Net net_input_mobile[n_mobile];
//     Net net_input_mnasnet[n_mnasnet];
//     Net net_input_inception[n_inception];
//     Net net_input_shuffle[n_shuffle];
//     Net net_input_resX[n_resX];

//     inputs.push_back(x);
//     auto x_ch0 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
//     auto x_ch1 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
//     auto x_ch2 = torch::unsqueeze(x2.index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
      
//     x_ch0.to(device);
//     x_ch1.to(device);
//     x_ch2.to(device);

//     auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(device);
//     inputs2.clear();
//     inputs2.push_back(x_cat);

//     for(int i=0;i<n_dense;i++){
//       net_input_dense[i].module = model[0].clone();
//       // if( i ==0 )
//       //   net_input_dense[i].module = model[0];
//       // else  net_input_dense[i].module = model[1];
//       net_input_dense[i].input = inputs;
//       net_input_dense[i].name = "DenseNet";
//       net_input_dense[i].index_n = i;
//       for(int j=0;j<WARMING;j++){
//           network_predict(&net_input_dense[i]);
//           cudaDeviceSynchronize();
//           net_input_dense[i].input = inputs;
//         }
//     }
//     // for(int j=0;j<WARMING;j++){
//     //       network_predict(&net_input_dense[1]);
//     //       cudaDeviceSynchronize();
//     //       net_input_dense[1].input = inputs;
//     //     }

//     for(int i=0;i<n_res;i++){
// 	  net_input_res[i].module = model[1];
//     net_input_res[i].name = "ResNet";
// 	  net_input_res[i].input = inputs;
//     net_input_res[i].index_n = i+n_dense;
//     for(int j=0;j<WARMING;j++){
//         network_predict(&net_input_res[i]);
//         cudaDeviceSynchronize();
//         net_input_res[i].input = inputs;
//       }
//   }

//   for(int i=0;i<n_alex;i++){
//     net_input_alex[i].module = model[2];
//     net_input_alex[i].index_n = i+ n_res + n_dense;
// 	  net_input_alex[i].input = inputs;
//     net_input_alex[i].name = "AlexNet";
//     for(int j=0;j<WARMING;j++){
//         network_predict(&net_input_alex[i]);
//         cudaDeviceSynchronize();
//         net_input_alex[i].input = inputs;
//       }
//   }

//   for(int i=0;i<n_vgg;i++){
//     net_input_vgg[i].module = model[3];
// 	  net_input_vgg[i].input = inputs;
//     net_input_vgg[i].name = "VGG";
//     net_input_vgg[i].index_n = i + n_alex + n_res + n_dense;
//     for(int j=0;j<WARMING;j++){
//         network_predict(&net_input_vgg[i]);
//         cudaDeviceSynchronize();
//         net_input_vgg[i].input = inputs;
//     }
//   }

//   for(int i=0;i<n_wide;i++){
//     net_input_wide[i].module = model[4];
// 	  net_input_wide[i].input = inputs;
//     net_input_wide[i].name = "WideResNet";
//     net_input_wide[i].index_n = i+n_alex + n_res + n_dense + n_vgg;
//     for(int j=0;j<WARMING;j++){
//         network_predict(&net_input_wide[i]);
//         cudaDeviceSynchronize();
//         net_input_wide[i].input = inputs;
//     }
//   }

//   for(int i=0;i<n_squeeze;i++){
//       net_input_squeeze[i].module = model[5];
//     net_input_squeeze[i].name = "SqueezeNet";
// 	  net_input_squeeze[i].input = inputs;
//     net_input_squeeze[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide;
//     for(int j=0;j<WARMING;j++){
//         network_predict(&net_input_squeeze[i]);
//         cudaDeviceSynchronize();
//         net_input_squeeze[i].input = inputs;
//     }
//   }

//   for(int i=0;i<n_mobile;i++){
//     net_input_mobile[i].module = model[6];
// 	  net_input_mobile[i].input = inputs;
//     net_input_mobile[i].name = "Mobile";
//     net_input_mobile[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze;
//     for(int j=0;j<WARMING;j++){
//         network_predict(&net_input_mobile[i]);
//         cudaDeviceSynchronize();
//         net_input_mobile[i].input = inputs;
//     }
//   }

//   for(int i=0;i<n_mnasnet;i++){
//     net_input_mnasnet[i].module = model[7];
// 	  net_input_mnasnet[i].input = inputs;
//     net_input_mnasnet[i].name = "MNASNet";
//     net_input_mnasnet[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile;
//     for(int j=0;j<WARMING;j++){
//         network_predict(&net_input_mnasnet[i]);
//         cudaDeviceSynchronize();
//         net_input_mnasnet[i].input = inputs;
//     }
//   }

//   for(int i=0;i<n_inception;i++){
//     net_input_inception[i].module = model[8];     
//     net_input_inception[i].input = inputs2;
//     net_input_inception[i].name = "Inception_v3";
//     net_input_inception[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet;
//     for(int j=0;j<WARMING;j++){
//         //cout<<j<<"!!!!!!!!!!!!\n";
//         network_predict(&net_input_inception[i]);
//         cudaDeviceSynchronize();
//         // net_input_inception[i].input = inputs2;
//     }
//   }

//   for(int i=0;i<n_shuffle;i++){
// 	  net_input_shuffle[i].module = model[9];
// 	  net_input_shuffle[i].input = inputs;
//     net_input_shuffle[i].name = "ShuffleNet";
//     net_input_shuffle[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception;
//     for(int j=0;j<WARMING;j++){
//         network_predict(&net_input_shuffle[i]);
//         cudaDeviceSynchronize();

//     }
//   }

//   for(int i=0;i<n_resX;i++){
//     net_input_resX[i].module = model[10];
// 	  net_input_resX[i].input = inputs;
//     net_input_resX[i].name = "ResNext";
//     net_input_resX[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle;
//     for(int j=0;j<WARMING;j++){
//         network_predict(&net_input_resX[i]);
//         cudaDeviceSynchronize();
//         net_input_resX[i].input = inputs;
//     }
//   }
//     //cudaProfilerStart();
//     cudaDeviceSynchronize();
//     double time1 = what_time_is_it_now();
//     //cudaProfilerStart();

//     for(int i=0;i<n_dense;i++){
//     if (pthread_create(&networkArray_dense[i], NULL, (void *(*)(void*))network_predict, &net_input_dense[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }
//   for(int i=0;i<n_res;i++){
//     if (pthread_create(&networkArray_res[i], NULL, (void *(*)(void*))network_predict, &net_input_res[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }
//   for(int i=0;i<n_alex;i++){
//     if (pthread_create(&networkArray_alex[i], NULL, (void *(*)(void*))network_predict, &net_input_alex[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }
//   for(int i=0;i<n_vgg;i++){
// 	  if (pthread_create(&networkArray_vgg[i], NULL, (void *(*)(void*))network_predict, &net_input_vgg[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }
//   for(int i=0;i<n_wide;i++){
//     if (pthread_create(&networkArray_wide[i], NULL, (void *(*)(void*))network_predict, &net_input_wide[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_squeeze;i++){
//     if (pthread_create(&networkArray_squeeze[i], NULL, (void *(*)(void*))network_predict, &net_input_squeeze[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_mobile;i++){
//     if (pthread_create(&networkArray_mobile[i], NULL, (void *(*)(void*))network_predict, &net_input_mobile[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_mnasnet;i++){
//     if (pthread_create(&networkArray_mnasnet[i], NULL, (void *(*)(void*))network_predict, &net_input_mnasnet[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }

//   for(int i=0;i<n_inception;i++){
//     if (pthread_create(&networkArray_inception[i], NULL, (void *(*)(void*))network_predict, &net_input_inception[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }
  
//   for(int i=0;i<n_shuffle;i++){
//     if (pthread_create(&networkArray_shuffle[i], NULL, (void *(*)(void*))network_predict, &net_input_shuffle[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }
//   for(int i=0;i<n_resX;i++){
//     if (pthread_create(&networkArray_resX[i], NULL, (void *(*)(void*))network_predict, &net_input_resX[i]) < 0){
//       perror("thread error");
//       exit(0);
//     }
//   }
//   // for(int i=0;i<n_inception;i++){
//   //   network_predict(&net_input_inception[i]);
//   // }

//   // predict_network(&net_input_inception[0]);
//   // predict_network(&net_input_inception[0]);
//   // predict_network(&net_input_inception[0]);
//   // predict_network(&net_input_inception[1]);
//   // predict_network(&net_input_inception[1]);
//   // predict_network(&net_input_inception[1]);
//   // cudaDeviceSynchronize();
//   // double time_1 = what_time_is_it_now();
//   // predict_network(&net_input_inception[0]);
//   // predict_network(&net_input_inception[1]);
//   // cudaDeviceSynchronize();
//   // double time_2 = what_time_is_it_now();
//   // std::cout << "\n!!!!!!!!!TOTAL EXECUTION TIME : "<<time_2-time_1<<"s ***** \n";
//   //  std::this_thread::sleep_for(5s);
//   //  g_controller.notify_all();
//   for (int i = 0; i < n_dense; i++){
//     pthread_join(networkArray_dense[i], NULL);
//   }
//   for (int i = 0; i < n_res; i++){
//     pthread_join(networkArray_res[i], NULL);
//   }
//   for (int i = 0; i < n_alex; i++){
//     pthread_join(networkArray_alex[i], NULL);
//   }
//   for (int i = 0; i < n_vgg; i++){
//     pthread_join(networkArray_vgg[i], NULL);
//   }
//   for (int i = 0; i < n_wide; i++){
//     pthread_join(networkArray_wide[i], NULL);
//   }
//   for (int i = 0; i < n_squeeze; i++){
//     pthread_join(networkArray_squeeze[i], NULL);
//   }
//   for (int i = 0; i < n_mobile; i++){
//     pthread_join(networkArray_mobile[i], NULL);
//   }
//   for (int i = 0; i < n_mnasnet; i++){
//     pthread_join(networkArray_mnasnet[i], NULL);
//   }
//   for (int i = 0; i < n_inception; i++){
//     pthread_join(networkArray_inception[i], NULL);
//   }
//   for (int i = 0; i < n_shuffle; i++){
//     pthread_join(networkArray_shuffle[i], NULL);
//   }
//   for (int i = 0; i < n_resX; i++){
//     pthread_join(networkArray_resX[i], NULL);
//   }
//   //cudaProfilerStop();
//   //cudaDeviceSynchronize();
//   double time2 = what_time_is_it_now();

// 	std::cout << "\n***** TOTAL EXECUTION TIME : "<<time2-time1<<"s ***** \n";

//   // std::cout<<"WITHOUT OVERHEAD TIME\n";

//   double tmp;
//   double max_end = end_time[0];
//   double min_start = start_time[0];
  
//   for(int i=0;i<n_all;i++){
//     min_start = min(min_start, start_time[i]);
//     cout<<"start = "<<end_time[i]-start_time[i]<<"\n";
//   }
//   for(int i=0;i<n_all;i++){
//     max_end = max(max_end,end_time[i]);
//    // cout<<"end = "<<end_time[i]<<"\n";
//   }
//   std::cout<<min_start<<" "<<max_end<<"\n";
//   std::cout<<max_end - min_start<<"\n";
// }