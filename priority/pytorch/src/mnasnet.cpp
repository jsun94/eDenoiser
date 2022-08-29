#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "mnasnet.h"

namespace F = torch::nn::functional;
//using namespace std;

void get_submodule_MNASNet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
	Dummy residual;
	for(auto children : module.named_children()){
		if(children.name == "layers"){
			for(auto child : children.value.named_children()){
				if(child.value.children().size() == 0){	//child.name == 0~7,14,15,16
					t_layer.layer = child.value;
					if(child.name == "2" || child.name == "5" || child.name == "16"){	//relu
						continue;
					}
					else if(child.name == "0" || child.name == "3" || child.name == "6" || child.name == "14"){
						t_layer.name = "conv";
					}
					else if(child.name == "1" || child.name == "4" || child.name == "15"){
						t_layer.name = "bn_relu";
					}
					else if(child.name == "7"){
						t_layer.name = "bn";
					}
					net.layers.push_back(t_layer);
				}
				else{	//child.name == 8~13
					for(auto block : child.value.named_children()){		//block.name == 0~3 (_inverted residual)
						for(auto inres : block.value.named_children()){		//inres.name == layers (layer in Inverted Residual)
							for(auto layer : inres.value.named_children()){		//layer.name == 0~7
								t_layer.layer = layer.value;
								if(layer.name == "2" || layer.name == "5"){		//relu
									continue;
								}
								else if(layer.name == "0" || layer.name == "3" || layer.name == "6"){
									t_layer.name = "conv";
								}
								else if(layer.name == "1" || layer.name == "4"){
									t_layer.name = "bn_relu";
								}
								else if(layer.name == "7"){
									t_layer.name = "bn";
								}
								net.layers.push_back(t_layer);	
							}
						}
						if(block.name != "0"){
							t_layer.layer = residual;
							t_layer.name = "Residual";
							t_layer.from_idx = {CURRENT_LAYERS, PREV_LAYERS};	//for residual
							net.layers.push_back(t_layer);
						}
					}
				}
			}
		}
		else if(children.name == "classifier"){
			for(auto child : children.value.named_children()){
				if(child.name == "0")	continue;	//dropout
				t_layer.layer = child.value;
				t_layer.name = "linear";
				net.layers.push_back(t_layer); 
			}
		}
	}
}

void *predict_warm_MNASNet(Net *mnasnet){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
		int i;
		float time;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		for(i=0;i<mnasnet->layers.size();i++){
			pthread_mutex_lock(&mutex_t[mnasnet->index_n]);
			cond_i[mnasnet->index_n] = 1;
			
			netlayer nl;
			nl.net = mnasnet;
			nl.net->index = i;

			th_arg th;
			th.arg = &nl;

			thpool_add_work(thpool,(void(*)(void *))forward_MNASNet,&th);

			while (cond_i[mnasnet->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[mnasnet->index_n], &mutex_t[mnasnet->index_n]);
			}

			i = mnasnet->index;
			mnasnet->input.clear();
			mnasnet->input.push_back(mnasnet->layers[i].output);
			pthread_mutex_unlock(&mutex_t[mnasnet->index_n]);
			/* FIFOQ 수정 -1 */
			cudaStreamSynchronize(streams[mnasnet->H_L][(mnasnet->index_s)%n_streamPerPool]);
		}

		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		std::cout << "\n*****"<<mnasnet->name<<" result  "<<time/1000<<"s ***** \n";
		std::cout << (mnasnet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void *predict_MNASNet(Net *mnasnet){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
		int i;
		int j;
		std::vector<torch::jit::IValue> origin_input = mnasnet->input;
		float time;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		for(j=0;j<total_dnn_iter;j++){
			mnasnet->input = origin_input;
		for(i=0;i<mnasnet->layers.size();i++){
			pthread_mutex_lock(&mutex_t[mnasnet->index_n]);
			cond_i[mnasnet->index_n] = 1;
			
			netlayer nl;
			nl.net = mnasnet;
			nl.net->index = i;

			th_arg th;
			th.arg = &nl;

			thpool_add_work(thpool,(void(*)(void *))forward_MNASNet,&th);

			while (cond_i[mnasnet->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[mnasnet->index_n], &mutex_t[mnasnet->index_n]);
			}

			i = mnasnet->index;
			mnasnet->input.clear();
			mnasnet->input.push_back(mnasnet->layers[i].output);
			pthread_mutex_unlock(&mutex_t[mnasnet->index_n]);
			/* FIFOQ 수정 -1 */
			cudaStreamSynchronize(streams[mnasnet->H_L][(mnasnet->index_s)%n_streamPerPool]);
		}
		}

		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		std::cout << "\n*****"<<mnasnet->name<<" result  "<<time/1000<<"s ***** \n";
		std::cout << (mnasnet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void forward_MNASNet(th_arg *th){
	{
/* FIFOQ 수정 -2 */
		at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]); // high, low
		pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
		netlayer *nl = th->arg;
		std::vector<torch::jit::IValue> inputs = nl->net->input;
		int k = nl->net->index;
		at::Tensor out;

		if(k == nl->net->gap){
			out = inputs[0].toTensor().mean({2,3});
			inputs.clear();
			inputs.push_back(out);
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
		}
		else if(nl->net->layers[k].name == "Residual"){
			int add_index = k + nl->net->layers[k].from_idx[0];
			out = nl->net->layers[add_index].output;
			for(int i=1;i<nl->net->layers[k].from_idx.size();i++){
				int add_index = k + nl->net->layers[k].from_idx[i];
				out += nl->net->layers[add_index].output;
			}
		}
		else{
			out = nl->net->layers[k].layer.forward(inputs).toTensor();

			if(k+1<nl->net->layers.size() && nl->net->layers[k+1].name == "bn"){
				nl->net->layers[k].output = out;
				k++;
				inputs.clear();
				inputs.push_back(out);
				out = nl->net->layers[k].layer.forward(inputs).toTensor();
			}
			else if(k+1<nl->net->layers.size() && nl->net->layers[k+1].name == "bn_relu"){
				nl->net->layers[k].output = out;
				k++;
				inputs.clear();
				inputs.push_back(out);
				out = nl->net->layers[k].layer.forward(inputs).toTensor();
				out = torch::relu(out);
			}
		}
		nl->net->layers[k].output = out;
		nl->net->index = k;
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
		pthread_mutex_unlock(&mutex_t[nl->net->index_n]);	
	}	
}

