#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "mobile.h"

namespace F = torch::nn::functional;


void get_submodule_mobilenet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
	for(auto children : module.named_children()){
		if(children.name == "features"){
			for(auto inverted : children.value.named_children()){	//inverted.name == 0~18
				int idx = net.layers.size();
				if(inverted.name == "0" || inverted.name == "18"){	//ConvBNReLU
					t_layer.layer = inverted.value;
					t_layer.name = "ConvBNReLU";
					net.layers.push_back(t_layer);
				}
				else{	//InvertedResidual
					for(auto block : inverted.value.named_children()){	//block.name == conv
						for(auto conv : block.value.named_children()){	//conv.name == 0~2
							t_layer.layer = conv.value;
							if(inverted.name == "1"){	//InvertedResidual_1
								if(conv.name == "0")	t_layer.name = "ConvBNReLU";
								else if(conv.name == "1")	t_layer.name = "conv";
								else if(conv.name == "2")	t_layer.name = "bn";
							}else{
								if(conv.name == "0" || conv.name == "1")	t_layer.name = "ConvBNReLU";
								else if(conv.name == "2")	t_layer.name = "conv";
								else if(conv.name == "3")	t_layer.name = "bn";
							}
							net.layers.push_back(t_layer);
						}
					}
					if(inverted.name == "3" || inverted.name == "5" || inverted.name == "6" || inverted.name == "8" || inverted.name == "9" || \
					inverted.name == "10" || inverted.name == "12" || inverted.name == "13" || inverted.name == "15" || inverted.name == "16"){ // use_res_connect = self.stride == 1 and inp == oup
						net.layers.back().name = "last_use_res_connect";
						net.layers[idx].name = "first_use_res_connect";
					}
				}
			}
		}
		else if(children.name == "classifier"){
			for(auto child : children.value.named_children()){
				if(child.name == "0") continue;	//dropout
				t_layer.layer = child.value;
				t_layer.name = "linear";
				net.layers.push_back(t_layer); 
			}
		}
	}
}

void *predict_warm_mobilenet(Net *mobile){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
		int i;
		float time;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		for(i=0;i<mobile->layers.size();i++){
			pthread_mutex_lock(&mutex_t[mobile->index_n]);
			cond_i[mobile->index_n] = 1;
			
			netlayer nl;
			nl.net = mobile;
			nl.net->index = i;

			th_arg th;
			th.arg = &nl;


			thpool_add_work(thpool,(void(*)(void *))forward_mobilenet,&th);
			while (cond_i[mobile->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[mobile->index_n], &mutex_t[mobile->index_n]);
			}
			i = mobile->index;
			mobile->input.clear();
			mobile->input.push_back(mobile->layers[i].output);
			pthread_mutex_unlock(&mutex_t[mobile->index_n]);
			/* FIFOQ 수정 -1 */
			cudaStreamSynchronize(streams[mobile->H_L][(mobile->index_s)%n_streamPerPool]);
		}

		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		std::cout << "\n*****"<<mobile->name<<" result " <<time/1000<<"s ***** \n";
		std::cout << "index num = "<< mobile->index_n << "	priority num = "<< mobile->priority << std::endl;
        std::cout << "Stream [" << mobile->H_L << "][" << (mobile->index_s)%n_streamPerPool <<"]" << std::endl;
		// std::cout << (mobile->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void *predict_mobilenet(Net *mobile){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
		int i;
		int j;
		std::vector<torch::jit::IValue> origin_input = mobile->input;
		float time;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		for(j=0;j<total_dnn_iter;j++){
			mobile->input = origin_input;
		for(i=0;i<mobile->layers.size();i++){
			pthread_mutex_lock(&mutex_t[mobile->index_n]);
			cond_i[mobile->index_n] = 1;
			
			netlayer nl;
			nl.net = mobile;
			nl.net->index = i;

			th_arg th;
			th.arg = &nl;


			thpool_add_work(thpool,(void(*)(void *))forward_mobilenet,&th);
			while (cond_i[mobile->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[mobile->index_n], &mutex_t[mobile->index_n]);
			}
			i = mobile->index;
			mobile->input.clear();
			mobile->input.push_back(mobile->layers[i].output);
			pthread_mutex_unlock(&mutex_t[mobile->index_n]);
			/* FIFOQ 수정 -1 */
			cudaStreamSynchronize(streams[mobile->H_L][(mobile->index_s)%n_streamPerPool]);
		}
		}

		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		std::cout << "\n*****"<<mobile->name<<" result " <<time/1000<<"s ***** \n";
		std::cout << "index num = "<< mobile->index_n << "	priority num = "<< mobile->priority << std::endl;
        std::cout << "Stream [" << mobile->H_L << "][" << (mobile->index_s)%n_streamPerPool <<"]" << std::endl;
		// std::cout << (mobile->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void forward_mobilenet(th_arg *th){
	{
/* FIFOQ 수정 -2 */
		at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]); // high, low
		pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
		netlayer *nl = th->arg;
		std::vector<torch::jit::IValue> inputs = nl->net->input;
		int k = nl->net->index;
		at::Tensor out;
		
		if(nl->net->layers[k].name == "first_use_res_connect"){
			nl->net->identity = inputs[0].toTensor();
		}

		// ***임시로 flatten 변수 사용***
		if(k == nl->net->flatten){	// nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
			out = torch::nn::functional::adaptive_avg_pool2d(inputs[0].toTensor(), \
			F::AdaptiveAvgPool2dFuncOptions(1)).reshape({inputs[0].toTensor().size(0), -1});
			inputs.clear();
			inputs.push_back(out);
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
		}

		/*
		InvertedResidual forward function

		def forward(self, x: Tensor) -> Tensor:
			if self.use_res_connect:
				return x + self.conv(x)
			else:
				return self.conv(x)
		*/
		else if(nl->net->layers[k].name == "last_use_res_connect"){
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			out = nl->net->identity + out;
		}
		else{
			out = nl->net->layers[k].layer.forward(inputs).toTensor();

			if(k+1 < nl->net->layers.size() && nl->net->layers[k+1].name == "bn"){
				nl->net->layers[k].output = out;
				k++;
				inputs.clear();
				inputs.push_back(out);
				out = nl->net->layers[k].layer.forward(inputs).toTensor();
			}
		}

		nl->net->layers[k].output = out;
		nl->net->index = k;
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
		pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
	}
}

