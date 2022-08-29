#ifndef TEST_H
#define TEST_H

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <pthread.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <signal.h>
#include <memory>
#include <stdlib.h>
#include <c10/cuda/CUDAFunctions.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sched.h>
#include <signal.h>
#include <nvToolsExt.h>

#include "thpool.h"
#include "alex.h"
#include "vgg.h"
#include "de_vgg.h"
#include "de_resnet.h"
#include "resnet.h"
#include "densenet.h"
#include "squeeze.h"
#include "mobile.h"
#include "mnasnet.h"
#include "inception.h"
#include "shuffle.h"
#include "efficient.h"
#include "regnet.h"
#include "de_regnet.h"
#include "origin_denoiser.h"
#include "custom_denoiser.h"
#include "decomposed_denoiser.h"
#include "co_resnet.h"
#include "de_inception.h"
#include "co_regnet.h"

/* FIFOQ 0 is PrioQ */
/* FIFOQ=0 -> Priority Q*/
/* FIFOQ=1 -> FIFOQ Q*/
#ifndef FIFOQ
#define FIFOQ 0
#endif

#define n_streamPerPool 32
#define n_Branch 3  //only use for branch

extern threadpool thpool; 
extern pthread_cond_t *cond_t;
extern pthread_mutex_t *mutex_t;
extern c10::DeviceIndex GPU_NUM;
extern int *cond_i;
extern std::vector<std::vector <at::cuda::CUDAStream>> streams;
extern double *start_min,*end_max;
extern at::Tensor h_out1;
extern at::Tensor h_out2;
extern at::Tensor h_out3;
extern at::Tensor h_out4;
extern at::Tensor cu_h_out1;
extern at::Tensor cu_h_out2;
extern at::Tensor cu_h_out3;
extern at::Tensor cu_h_out4;
extern at::Tensor de_h_out1;
extern at::Tensor de_h_out2;
extern at::Tensor de_h_out3;
extern at::Tensor de_h_out4;
extern int total_iter;
extern int total_dnn_iter;
double what_time_is_it_now();

#endif
