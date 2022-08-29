#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include <thread>
#include <unistd.h>
#include "origin_denoiser.h"

/*

event_idx : branch_num in inception (for recording event)
input_idx : the index of the input from the current layer
skip : Number of layer modules in one branch (How many more signals do thread have to send)
branch_idx : The last layer index of the branch to determine if the operation is complete(exe_success)

*/

namespace F = torch::nn::functional;
using namespace std;

void get_submodule_origin_denoiser(torch::jit::script::Module module, Net &net){
	Layer t_layer;
    if(module.children().size() == 0){ 
        t_layer.layer = module;
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		get_submodule_origin_denoiser(children.value, net);
	}
}

void *predict_warm_origin_denoiser(Net *or_denoiser){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
        int i;
        float time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);

        for(i=0;i<or_denoiser->layers.size();i++){
            pthread_mutex_lock(&mutex_t[or_denoiser->index_n]);
            cond_i[or_denoiser->index_n] = 1;
            // or_denoiser->layers[i].exe_success = false;

            netlayer nl;
            nl.net = or_denoiser;
            nl.net->index = i;

            th_arg th;
            th.arg = &nl;
            // std::cout<<"layer index : "<<i<<" name : "<<nl.net->layers[i].name<<std::endl;
            thpool_add_work(thpool,(void(*)(void *))forward_origin_denoiser,&th);
            
            while (cond_i[or_denoiser->index_n] == 1)
            {
                pthread_cond_wait(&cond_t[or_denoiser->index_n], &mutex_t[or_denoiser->index_n]);
            }
            i = nl.net->index;
            or_denoiser->input.clear();
            or_denoiser->input.push_back(or_denoiser->layers[i].output);
            pthread_mutex_unlock(&mutex_t[or_denoiser->index_n]);
            /* FIFOQ 수정 -1 밑에 네줄*/
            // cudaStreamSynchronize(streams[or_denoiser->H_L][(or_denoiser->index_s)%n_streamPerPool]);
        }
        cudaStreamSynchronize(streams[or_denoiser->H_L][(or_denoiser->index_s)%n_streamPerPool]);

        // cudaStreamSynchronize(streams[or_denoiser->stream_id[0]]);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        std::cout << "\n*****"<<or_denoiser->name<<" result  "<<time/1000<<"s ***** \n";
        std::cout << "index num = "<< or_denoiser->index_n << "	priority num = "<< or_denoiser->priority << std::endl;
        std::cout << "Stream [" << or_denoiser->H_L << "][" << (or_denoiser->index_s)%n_streamPerPool <<"]" << std::endl;
        // std::cout << (or_denoiser->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}


void *predict_origin_denoiser(Net *or_denoiser){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
        int i;
		int j;	/* for several inputs */
        float time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
		std::vector<torch::jit::IValue> origin_input = or_denoiser->input;	/* for several inputs */

		for(j = 0; j < total_iter; j++){	/* for several inputs */
			or_denoiser->input = origin_input;	/* for several inputs */
        for(i=0;i<or_denoiser->layers.size();i++){
            pthread_mutex_lock(&mutex_t[or_denoiser->index_n]);
            cond_i[or_denoiser->index_n] = 1;
            // or_denoiser->layers[i].exe_success = false;

            netlayer nl;
            nl.net = or_denoiser;
            nl.net->index = i;

            th_arg th;
            th.arg = &nl;
            // std::cout<<"layer index : "<<i<<" name : "<<nl.net->layers[i].name<<std::endl;
            thpool_add_work(thpool,(void(*)(void *))forward_origin_denoiser,&th);
            
            while (cond_i[or_denoiser->index_n] == 1)
            {
                pthread_cond_wait(&cond_t[or_denoiser->index_n], &mutex_t[or_denoiser->index_n]);
            }
            i = nl.net->index;
            or_denoiser->input.clear();
            or_denoiser->input.push_back(or_denoiser->layers[i].output);
            pthread_mutex_unlock(&mutex_t[or_denoiser->index_n]);
            /* FIFOQ 수정 -1 밑에 네줄*/
            // cudaStreamSynchronize(streams[or_denoiser->H_L][(or_denoiser->index_s)%n_streamPerPool]);
            }
		}	/* for several inputs */
        cudaStreamSynchronize(streams[or_denoiser->H_L][(or_denoiser->index_s)%n_streamPerPool]);

        //cudaStreamSynchronize(streams[or_denoiser->stream_id[0]]);
        //cudaEventRecord(end, streams[or_denoiser->stream_id[0]]);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        std::cout << "\n*****"<<or_denoiser->name<<" result  "<<time/1000<<"s ***** \n";
        std::cout << "index num = "<< or_denoiser->index_n << "	priority num = "<< or_denoiser->priority << std::endl;
	    std::cout << "Stream [" << or_denoiser->H_L << "][" << (or_denoiser->index_s)%n_streamPerPool <<"]" << std::endl;
        // std::cout << (or_denoiser->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void forward_origin_denoiser(th_arg *th){
	{
/* FIFOQ 수정 -2 */
        at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]); // high, low
		//at::cuda::CUDAStreamGuard guard(streams[th->arg->net->stream_id[0]]);
        pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
        
        char str[30];
        sprintf(str, "or_denoiser layer - %d", th->arg->net->index);
        nvtxRangeId_t id1 = nvtxRangeStartA(str);

        netlayer *nl = th->arg;
        int k = nl->net->index;
		// std::cout << "k : " << k << std::endl;
        int n_all = nl->net->n_all;
        std::vector<torch::jit::IValue> inputs;
		inputs = nl->net->input;

        // pthread_mutex_unlock(&mutex_t[nl->net->index_n]); 
        at::Tensor out;
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
		if(k == 5){
			h_out1 = out;
		}
		else if (k == 14){
			h_out2 = out;
		}
		else if (k == 23){
			h_out3 = out;
		}
		else if (k == 32){
			h_out4 = out;
		}
		if(k == 42){
			out = torch::cat({out, h_out4}, 1);
		}
		if(k == 52){
			out = torch::cat({out, h_out3},1);
		}
		if(k == 62){
			out = torch::cat({out, h_out2},1);
		}
		if(k == 72){
			out = torch::cat({out, h_out1},1);
		}

        nl->net->layers[k].output = out;
     

        nvtxRangeEnd(id1);

        // pthread_mutex_lock(&mutex_t[nl->net->index_n]);
		//jsh
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
		//jsh
        pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
    }
}
