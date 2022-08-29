#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "regnet.h"

namespace F = torch::nn::functional;

void get_submodule_regnet(torch::jit::script::Module module,Net &net){
    Layer t_layer;
    Dummy residual;
    for(auto children : module.named_children()){
        if(children.value.children().size() == 0){  //avgpool, fc
            t_layer.layer = children.value;
            t_layer.name = children.name;
            net.layers.push_back(t_layer);
        }else{
            if(children.name == "stem"){
                t_layer.input_idx = 0;
                t_layer.layer = children.value;
                t_layer.name = "ConvBnRelu";
                net.layers.push_back(t_layer);
            }
            else{   //children.name == "trunk_output"
                for(auto block : children.value.named_children()){  //block1,2,3,4
                    int is_first = 0;
                    for(auto anystage : block.value.named_children()){  //blockn-m
                        for(auto resblock : anystage.value.named_children()){   //proj, f, activation
                            if(resblock.name == "proj"){
                                is_first++;
                                t_layer.input_idx = 0;
                                t_layer.layer = resblock.value;
                                t_layer.name = "ConvBn";
                                net.layers.push_back(t_layer);
                            }else if(resblock.name == "activation"){
                                t_layer.layer = residual;
                                t_layer.name = "Residual";
                                t_layer.from_idx = {CURRENT_LAYERS, REG_PREV_LAYERS};
                                net.layers.push_back(t_layer);
                            }
                            else{
                                for(auto in_block : resblock.value.named_children()){   //a,b,se,c
                                    if(in_block.name == "se"){
                                        for(auto in_se : in_block.value.named_children()){
                                            t_layer.layer = in_se.value;
                                            if(in_se.name == "avgpool")    t_layer.name = "avgpool";
                                            else if(in_se.name == "fc1")    t_layer.name = "ConvReLU";
                                            else if(in_se.name == "fc2")    t_layer.name = "ConvSigmoid";
                                            else if(in_se.name == "activation" || in_se.name == "scale_activation") continue;
                                            net.layers.push_back(t_layer);
                                        }
                                    }else{  //a,b,c
                                        if(is_first == 1){
                                            t_layer.input_idx = -2;
                                            is_first++;
                                        }
                                        t_layer.layer = in_block.value;
                                        t_layer.name = "ConvBnRelu";
                                        net.layers.push_back(t_layer);
                                        t_layer.input_idx = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void *predict_warm_regnet(Net *regnet){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
		int i;
		float time;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		for(i=0;i<regnet->layers.size();i++){
			pthread_mutex_lock(&mutex_t[regnet->index_n]);
			cond_i[regnet->index_n] = 1;
			netlayer nl;
			nl.net = regnet;
			nl.net->index = i;

			th_arg th;
			th.arg = &nl;

			thpool_add_work(thpool,(void(*)(void *))forward_regnet,&th);
			
            while (cond_i[regnet->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[regnet->index_n], &mutex_t[regnet->index_n]);
			}
			i = regnet->index;
			regnet->input.clear();
			regnet->input.push_back(regnet->layers[i].output);
			pthread_mutex_unlock(&mutex_t[regnet->index_n]);
/* FIFOQ 수정-1 */
            cudaStreamSynchronize(streams[regnet->H_L][(regnet->index_s)%n_streamPerPool]);
		}

		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
        
		std::cout << "\n*****"<<regnet->name<<" result " <<time/1000<<"s ***** \n";
		std::cout << "index num = "<< regnet->index_n << "	priority num = "<< regnet->priority << std::endl;
        std::cout << "Stream [" << regnet->H_L << "][" << (regnet->index_s)%n_streamPerPool <<"]" << std::endl;
		// std::cout << (regnet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void *predict_regnet(Net *regnet){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
		int i;
		int j;
		float time;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		std::vector<torch::jit::IValue> origin_input = regnet->input; /* for several inputs */
		for(j = 0; j < total_iter; j++){	/* for several inputs */
			regnet->input = origin_input;	/* for several inputs */
		for(i=0;i<regnet->layers.size();i++){
			pthread_mutex_lock(&mutex_t[regnet->index_n]);
			cond_i[regnet->index_n] = 1;
			netlayer nl;
			nl.net = regnet;
			nl.net->index = i;

			th_arg th;
			th.arg = &nl;

			thpool_add_work(thpool,(void(*)(void *))forward_regnet,&th);
			
            while (cond_i[regnet->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[regnet->index_n], &mutex_t[regnet->index_n]);
			}
			i = regnet->index;
			regnet->input.clear();
			regnet->input.push_back(regnet->layers[i].output);
			pthread_mutex_unlock(&mutex_t[regnet->index_n]);
/* FIFOQ 수정-1 */
            cudaStreamSynchronize(streams[regnet->H_L][(regnet->index_s)%n_streamPerPool]);
		}
		}
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
        
		std::cout << "\n*****"<<regnet->name<<" result " <<time/1000<<"s ***** \n";
		std::cout << "index num = "<< regnet->index_n << "	priority num = "<< regnet->priority << std::endl;
        std::cout << "Stream [" << regnet->H_L << "][" << (regnet->index_s)%n_streamPerPool <<"]" << std::endl;
		// std::cout << (regnet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void forward_regnet(th_arg *th){
    {
		at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]);
		pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
		netlayer *nl = th->arg;
		std::vector<torch::jit::IValue> inputs/* = nl->net->input*/;
		int k = nl->net->index;


		// char str[30];
		// sprintf(str, "Reg layer - %d", k);
		// nvtxRangeId_t id1 = nvtxRangeStartA(str);
		
        at::Tensor out;
		cudaEvent_t start, end;
		float l_time;
		// cudaEventCreate(&start);
		// cudaEventCreate(&end);
		// cudaEventRecord(start);
        if(nl->net->layers[k].input_idx != 0){
            inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
        }else{
            inputs = nl->net->input;
        }

		if(nl->net->layers[k].name == "avgpool"){
            nl->net->identity = inputs[0].toTensor();
            out = nl->net->layers[k].layer.forward(inputs).toTensor();

            if(k+1 < nl->net->layers.size()){
                nl->net->layers[k].output = out;
                k++;
                inputs.clear();
                inputs.push_back(out);
            }
        }
        if(nl->net->layers[k].name == "Residual"){
            int add_index = k + nl->net->layers[k].from_idx[0];
			out = nl->net->layers[add_index].output;
			for(int i=1;i<nl->net->layers[k].from_idx.size();i++){
				int add_index = k + nl->net->layers[k].from_idx[i];
				out += nl->net->layers[add_index].output;
			}
            out = F::relu(out);
        }
        else if(k==nl->net->flatten){	//flatten
            out = inputs[0].toTensor().view({nl->net->layers[k-1].output.size(0), -1});
            inputs.clear();
            inputs.push_back(out);
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
		}
        else{
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
            if(nl->net->layers[k].name == "ConvReLU"){
                out = F::relu(out);
            }
            else if(nl->net->layers[k].name == "ConvSigmoid"){
                out = torch::sigmoid(out);
                out = nl->net->identity * out;
            }
        }

		// cudaStreamSynchronize(streams[th->arg->net->stream_id[0]]);

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