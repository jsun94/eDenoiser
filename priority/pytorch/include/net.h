#ifndef NET_H
#define NET_H

#include <vector>
#include <torch/torch.h>
#include <string>
#include <functional>
#include "cuda_runtime.h"

#include "curand.h"
#include "cublas_v2.h"

struct Dummy : torch::jit::Module{};

typedef struct _layer
{
	at::Tensor output;
	std::string name;	//layer name
	torch::jit::Module layer;
	bool exe_success;	//layer operation complete or not
	std::vector<int> from_idx;	//concat
	std::vector<int> branch_idx;	// last layer idx of branch for eventrecord
	int input_idx; 	//network with branch
	int event_idx;	//network with branch
	int skip;	//inception skip num in a branch
	int stream_idx;	//stream index of current layers
}Layer;

typedef struct _net
{
	std::vector<Layer> layers;
	std::vector<torch::jit::IValue> input;
	at::Tensor identity;	//resnet
	std::vector<cudaEvent_t> record;
	std::vector<at::Tensor> chunk; //shuffle
	std::string name;	//network name
	//std::vector<int> stream_id;		//stream index, index of stream for branch layer 
	//nvtxDomainHandle_t domain;	//nvtx annotation
	int index; //layer index
	int index_n; //network index
	int index_s; // stream index
	int index_b; // index of stream for branch layer 
	int n_all; // all network num
	int flatten; //flatten layer index
	int priority; // priQ priority
	int H_L; // 0 = HIGH , 1 = LOW
	int gap; //global average pooling index(mnas, shuffle)
	FILE *fp;
}Net;

typedef struct _netlayer
{
	Net *net;
}netlayer;

#endif
