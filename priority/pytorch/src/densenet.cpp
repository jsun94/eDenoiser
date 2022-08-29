#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <cuda_profiler_api.h>
#include <inttypes.h>
#include <memory>
#include <nvToolsExt.h>
#include <stdlib.h>
#include <string.h>
#include "densenet.h"

using namespace std;
namespace F = torch::nn::functional;

void get_submodule_densenet(torch::jit::script::Module module,Net &net){
    Layer t_layer;
    Dummy concat;
	for(auto child : module.named_children()){
        if(child.value.children().size()==0){ //child.name == classifier
			t_layer.layer = child.value;
            if(child.name == "classifier") t_layer.name = "classifier";
			net.layers.push_back(t_layer);
		}else{  //child.name == features
            for(auto block : child.value.named_children()){    //block.name == conv0, norm0, relu0, pool, denseblock, transition
				if(child.name == "features"){
                    if(block.value.children().size() == 0){    //conv0 ,norm0 , relu0, pool 
                        t_layer.layer = block.value;
                        if(block.name == "conv0") t_layer.name = "conv";
                        else if(block.name == "norm0" || block.name == "norm5") t_layer.name = "norm_relu";
                        else if(block.name == "relu0") continue;
                        else if(block.name == "pool0") t_layer.name = "pool";
			            net.layers.push_back(t_layer);
                    
                    }else{  //block.name == denseblock, Transition
                        for(auto layer : block.value.named_children()){
                            if(layer.value.children().size() == 0){  // layer.name == transition (layers in transition block)
                                t_layer.layer = layer.value;
                                if(layer.name == "conv") t_layer.name = "conv";
                                else if(layer.name == "norm") t_layer.name = "norm_relu";
                                else if(layer.name == "relu") continue;
                                else if(layer.name == "pool") t_layer.name = "pool";
                                net.layers.push_back(t_layer);
                            }else{  //layer.name == denselayer
                                t_layer.from_idx = {CURRENT_DENSELAYER};
                                t_layer.layer = concat;
                                t_layer.name = "concat";
                                net.layers.push_back(t_layer);
                                for(auto in_layer : layer.value.named_children()){  //layers in denselayer
                                    t_layer.from_idx.clear();
                                    t_layer.layer = in_layer.value;
                                    if(in_layer.name == "conv1" || in_layer.name == "conv2") t_layer.name = "conv";
                                    else if(in_layer.name == "norm1" || in_layer.name == "norm2") t_layer.name = "norm_relu";
                                    else if(in_layer.name == "relu1" || in_layer.name == "relu2") continue;
                                    net.layers.push_back(t_layer);
                                }
                                t_layer.from_idx = {PREV_DENSELAYER, CURRENT_DENSELAYER};   //for concat
                                t_layer.layer = concat;
                                t_layer.name = "concat";
                                net.layers.push_back(t_layer);
                            }
                        }
                    }
                }
            }
        }
    }
}

void *predict_warm_densenet(Net *densenet){
    {
        //at::cuda::set_device(1);
        at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
        int i;
        cudaEvent_t start, end;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        //start_min[densenet->index_n] = what_time_is_it_now();
        for(i=0;i<densenet->layers.size();i++){
            pthread_mutex_lock(&mutex_t[densenet->index_n]);
            cond_i[densenet->index_n] = 1; 
            netlayer nl;
            nl.net = densenet;
            nl.net->index = i;

            th_arg th;
            th.arg = &nl;

            thpool_add_work(thpool,(void(*)(void *))forward_densenet,&th);

            while (cond_i[densenet->index_n] == 1)
            {
                pthread_cond_wait(&cond_t[densenet->index_n], &mutex_t[densenet->index_n]);
            }
            i = nl.net->index;
            densenet->input.clear();
            densenet->input.push_back(densenet->layers[i].output);
            pthread_mutex_unlock(&mutex_t[densenet->index_n]);
            /* FIFOQ 수정 -1 */
            cudaStreamSynchronize(streams[densenet->H_L][(densenet->index_s)%n_streamPerPool]);
            
        }

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
    
        std::cout << "\n*****"<<densenet->name<<" result  "<<time/1000<<"s ***** \n";
        std::cout << "index num = "<< densenet->index_n << "	priority num = "<< densenet->priority << std::endl;
        std::cout << "Stream [" << densenet->H_L << "][" << (densenet->index_s)%n_streamPerPool <<"]" << std::endl;
        // std::cout << (densenet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
    }
}

void *predict_densenet(Net *densenet){
    {
        //at::cuda::set_device(1);
        at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
        int i;
		int j;
        cudaEvent_t start, end;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
		std::vector<torch::jit::IValue> origin_input = densenet->input; /* for several input */
        cudaEventRecord(start);
        //start_min[densenet->index_n] = what_time_is_it_now();
		for(j=0; j<total_dnn_iter; j++){
			densenet->input = origin_input;
        for(i=0;i<densenet->layers.size();i++){
            pthread_mutex_lock(&mutex_t[densenet->index_n]);
            cond_i[densenet->index_n] = 1; 
            netlayer nl;
            nl.net = densenet;
            nl.net->index = i;

            th_arg th;
            th.arg = &nl;

            thpool_add_work(thpool,(void(*)(void *))forward_densenet,&th);

            while (cond_i[densenet->index_n] == 1)
            {
                pthread_cond_wait(&cond_t[densenet->index_n], &mutex_t[densenet->index_n]);
            }
            i = nl.net->index;
            densenet->input.clear();
            densenet->input.push_back(densenet->layers[i].output);
            pthread_mutex_unlock(&mutex_t[densenet->index_n]);
            /* FIFOQ 수정 -1 */
            cudaStreamSynchronize(streams[densenet->H_L][(densenet->index_s)%n_streamPerPool]);
            
        }
		}

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
    
        std::cout << "\n*****"<<densenet->name<<" result  "<<time/1000<<"s ***** \n";
        std::cout << "index num = "<< densenet->index_n << "	priority num = "<< densenet->priority << std::endl;
        std::cout << "Stream [" << densenet->H_L << "][" << (densenet->index_s)%n_streamPerPool <<"]" << std::endl;
        // std::cout << (densenet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
    }
}

void forward_densenet(th_arg *th){
    {
/* FIFOQ 수정 -2 */
        at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]); // high, low
        pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
        netlayer *nl = th->arg;
        std::vector<torch::jit::IValue> inputs = nl->net->input;
        int k =nl->net->index;
        // char str[30];
        // sprintf(str, "Dense layer - %d", k);
        // nvtxRangeId_t id1 = nvtxRangeStartA(str);
        at::Tensor out;
        cudaEvent_t l_start, l_end;
        float l_time;
        //cudaEventCreate(&l_start);
        //cudaEventCreate(&l_end);
        //cudaEventRecord(l_start);
        //at::cuda::setCurrentCUDAStream(streams[nl->net->H_L][(nl->net->index_s)%(n_streamPerPool)]);
        //std::cout<<"dense device num in stream = "<<c10::cuda::current_device()<<"\n";
        if(k == nl->net->flatten){
            out = F::adaptive_avg_pool2d(inputs[0].toTensor(), F::AdaptiveAvgPool2dFuncOptions(1));
            out = out.view({out.size(0), -1});
            inputs.clear();
            inputs.push_back(out);
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
        }
        else if(nl->net->layers[k].name == "concat"){
            std::vector<at::Tensor> cat_input;
            for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
                cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
            }
            out = torch::cat(cat_input, 1);
        }
        else{
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
            if(nl->net->layers[k].name == "norm_relu"){
                out = torch::relu(out);
            }
            if(k+1 < nl->net->layers.size() && nl->net->layers[k+1].name == "norm_relu"){
                nl->net->layers[k].output = out;
                k++;
                inputs.clear();
                inputs.push_back(out);
                out = nl->net->layers[k].layer.forward(inputs).toTensor();
                out = torch::relu(out);
            }
            if(k+1 < nl->net->layers.size() && nl->net->layers[k+1].name == "conv"){
				nl->net->layers[k].output = out;
				k++;
				inputs.clear();
                inputs.push_back(out);
				out = nl->net->layers[k].layer.forward(inputs).toTensor();
			}
        }
        cudaStreamSynchronize(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]);
        nl->net->layers[k].output = out;
        nl->net->index = k;
	    cond_i[nl->net->index_n]=0;
	    // nvtxRangeEnd(id1);
        //cudaEventRecord(l_end);
        //cudaEventSynchronize(l_end);
        //cudaEventElapsedTime(&l_time, l_start, l_end);
        //fprintf((nl->net->fp),"%d,%lf\n",nl->net->index,l_time/1000);
        //std::cout<<"forward end\n\n";
        pthread_cond_signal(&cond_t[nl->net->index_n]);
	    pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
    }
}