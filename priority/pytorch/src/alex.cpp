#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include "cuda_runtime.h"
#include "alex.h"

namespace F = torch::nn::functional;

void get_submodule_alexnet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
	for(auto child : module.named_children()){
		if(child.value.children().size()==0){	//avgpool
			t_layer.layer = child.value;
			t_layer.name = "avgpool";
			net.layers.push_back(t_layer);
		}
		else{	//feature , classifier
			for(auto ch : child.value.named_children()){
				if(child.name == "features"){
					t_layer.layer = ch.value;
					if(ch.name == "0" || ch.name == "3" || ch.name == "6" || ch.name == "8" || ch.name == "10"){
						t_layer.name = "conv";
					}
					else if(ch.name == "1" || ch.name == "4" || ch.name == "7" || ch.name == "9" || ch.name == "11"){
						t_layer.name = "relu";
					}
					else if(ch.name == "2" || ch.name == "5" || ch.name == "12"){
						t_layer.name = "maxpool";
					}
					net.layers.push_back(t_layer);
				}
				else if(child.name == "classifier"){
					t_layer.layer = ch.value;
					if(ch.name == "0" || ch.name == "3" ) continue;		//dropout
					else if(ch.name == "1" || ch.name == "4" || ch.name == "6"){
						t_layer.name = "linear";
					}
					else if(ch.name == "2" || ch.name == "5"){
						t_layer.name = "relu";
					}
					net.layers.push_back(t_layer);
				}
			}
		}
	}
}

void *predict_warm_alexnet(Net *alex){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
/* FIFOQ 수정 -1 */
		std::vector<torch::jit::IValue> inputs = alex->input;
		int i;
		//double time1 = what_time_is_it_now();
		float time;
		sigval sig;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		for(i=0;i<alex->layers.size();i++){
			pthread_mutex_lock(&mutex_t[alex->index_n]);
			cond_i[alex->index_n] = 1;
			
			netlayer nl;
			nl.net = alex;
			nl.net->index = i;
			
			//필요해?
			th_arg th;
			th.arg = &nl;

			thpool_add_work(thpool,(void(*)(void *))forward_alexnet,&th);

			while (cond_i[alex->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[alex->index_n], &mutex_t[alex->index_n]);
			}
			i = nl.net->index;
			alex->input.clear();
			alex->input.push_back(alex->layers[i].output);
			pthread_mutex_unlock(&mutex_t[alex->index_n]);
			cudaStreamSynchronize(streams[alex->H_L][(alex->index_s)%n_streamPerPool]);
		}
/* FIFOQ 수정 -2 */
		
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		//double time2 = what_time_is_it_now();

		std::cout << "\n*****"<<alex->name<<" result  "<<time/10000<<"s ***** \n";
		std::cout << "index num = "<< alex->index_n << "	priority num = "<< alex->priority << std::endl;
        std::cout << "Stream [" << alex->H_L << "][" << (alex->index_s)%n_streamPerPool <<"]" << std::endl;
		// std::cout << (alex->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void *predict_alexnet(Net *alex){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
/* FIFOQ 수정 -1 */
		std::vector<torch::jit::IValue> inputs = alex->input;
		int i;
		int j;
		//double time1 = what_time_is_it_now();
		float time;
		sigval sig;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		std::vector<torch::jit::IValue> origin_input = alex->input; /* for several inputs */
		cudaEventRecord(start);
		for(j=0; j<total_dnn_iter; j++){
			alex->input = origin_input;
		for(i=0;i<alex->layers.size();i++){
			pthread_mutex_lock(&mutex_t[alex->index_n]);
			cond_i[alex->index_n] = 1;
			
			netlayer nl;
			nl.net = alex;
			nl.net->index = i;
			
			//필요해?
			th_arg th;
			th.arg = &nl;

			thpool_add_work(thpool,(void(*)(void *))forward_alexnet,&th);

			while (cond_i[alex->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[alex->index_n], &mutex_t[alex->index_n]);
			}
			i = nl.net->index;
			alex->input.clear();
			alex->input.push_back(alex->layers[i].output);
			pthread_mutex_unlock(&mutex_t[alex->index_n]);
			cudaStreamSynchronize(streams[alex->H_L][(alex->index_s)%n_streamPerPool]);
		}
		}
/* FIFOQ 수정 -2 */
		
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
		//double time2 = what_time_is_it_now();

		std::cout << "\n*****"<<alex->name<<" result  "<<time/10000<<"s ***** \n";
		std::cout << "index num = "<< alex->index_n << "	priority num = "<< alex->priority << std::endl;
        std::cout << "Stream [" << alex->H_L << "][" << (alex->index_s)%n_streamPerPool <<"]" << std::endl;
		// std::cout << (alex->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void forward_alexnet(th_arg *th){
	{
/* FIFOQ 수정 -3 */
		at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]); // high, low
		pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
		netlayer *nl = th->arg;
		std::vector<torch::jit::IValue> inputs = nl->net->input;
		int k = nl->net->index;
		// char str[30];
		// sprintf(str, "Alex layer - %d", k);
		// nvtxRangeId_t id1 = nvtxRangeStartA(str);
		at::Tensor out;
		cudaEvent_t start, end;
		float l_time;
		// cudaEventCreate(&start);
		// cudaEventCreate(&end);
		// cudaEventRecord(start);

		if(k==nl->net->flatten){	//flatten
				out = inputs[0].toTensor().view({nl->net->layers[k-1].output.size(0), -1});
				inputs.clear();
				inputs.push_back(out);
		}
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
		if(k+1 < nl->net->layers.size() && nl->net->layers[k+1].name == "relu"){
			nl->net->layers[k].output = out;
			k++;
			inputs.clear();
			inputs.push_back(out);
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
		}
/* FIFOQ 수정 -4  밑에줄 아마 필요없을거임 그래서 지움*/
		//cudaStreamSynchronize(streams[th->arg->net->stream_id[0]]); // 나중에 지워야함
		nl->net->layers[k].output = out;
		nl->net->index = k;
		cond_i[nl->net->index_n]=0;
		// nvtxRangeEnd(id1);
		// cudaEventRecord(end);
		// cudaEventSynchronize(end);
		// cudaEventElapsedTime(&l_time, start, end);
		//fprintf((nl->net->fp),"%d,%lf\n",nl->net->index,l_time/1000);
		pthread_cond_signal(&cond_t[nl->net->index_n]);
		pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
	}
}

