#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "efficient.h"

namespace F = torch::nn::functional;

void get_submodule_efficientnet(torch::jit::script::Module module,Net &net){
    Layer t_layer;
    Dummy residual;
	for(auto children : module.named_children()){
        if(children.value.children().size() == 0){     //avgpool
            t_layer.layer = children.value;
            t_layer.from_idx = {-1};
            t_layer.name = "avgpool";
            net.layers.push_back(t_layer);
        }
        else{   //children.name: "features", "classifier"
            if(children.name == "features"){
                for(auto child : children.value.named_children()){  //child.name: 0,1,..,8
                    if(child.name == "0" || child.name == "8"){
                        t_layer.layer = child.value;
                        t_layer.from_idx = {-1};
                        t_layer.name = "ConvBnSiLU";
                        net.layers.push_back(t_layer);
                    }
                    else{
                        for(auto MBConv : child.value.named_children()){     //MBConv: 0,1,..
                            for(auto block : MBConv.value.named_children()){    //block: "block", "stochastic_depth"
                                if(block.name == "block"){
                                    if(child.name == "1"){
                                        for(auto in_block : block.value.named_children()){
                                            t_layer.layer = in_block.value;
                                            if(in_block.name == "0"){
                                                t_layer.name = "ConvBnSiLU";
                                                t_layer.from_idx = {-1};
                                                net.layers.push_back(t_layer);
                                            }
                                            else if(in_block.name == "1"){
                                                for(auto layer : in_block.value.named_children()){
                                                    t_layer.layer = layer.value;
                                                    if(layer.name == "avgpool"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -2};
                                                        t_layer.name = "avgpool";
                                                    }
                                                    else if(layer.name == "fc1"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -3};
                                                        t_layer.name = "ConvSiLU";
                                                    }
                                                    else if(layer.name == "fc2"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -4};
                                                        t_layer.name = "ConvSigmoid";
                                                    }
                                                    else if(layer.name == "activation" || layer.name == "scale_activation") continue;
                                                    net.layers.push_back(t_layer);
                                                }
                                            }
                                            else if(in_block.name == "2"){
                                                if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                else    t_layer.from_idx = {-1, -5};
                                                t_layer.name = "ConvBn";
                                                net.layers.push_back(t_layer);
                                            }
                                        }
                                        if(MBConv.name != "0"){
                                            t_layer.layer = residual;
                                            t_layer.name = "Residual";
                                            t_layer.from_idx = {CURRENT_LAYERS, PREV_LAYERS_1};
                                            net.layers.push_back(t_layer);
                                        }
                                    }
                                    else{   //child.name: 2,3,..,7
                                        for(auto in_block : block.value.named_children()){
                                            t_layer.layer = in_block.value;
                                            if(in_block.name == "0" || in_block.name == "1"){
                                                if(in_block.name == "0" || MBConv.name == "0")    t_layer.from_idx = {-1};
                                                else    t_layer.from_idx = {-1, -2};
                                                t_layer.name = "ConvBnSiLU";
                                                net.layers.push_back(t_layer);
                                            }
                                            else if(in_block.name == "2"){
                                                for(auto layer : in_block.value.named_children()){
                                                    t_layer.layer = layer.value;
                                                    if(layer.name == "avgpool"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -3};
                                                        t_layer.name = "avgpool";
                                                    }
                                                    else if(layer.name == "fc1"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -4};
                                                        t_layer.name = "ConvSiLU";
                                                    }
                                                    else if(layer.name == "fc2"){
                                                        if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                        else    t_layer.from_idx = {-1, -5};
                                                        t_layer.name = "ConvSigmoid";
                                                    }
                                                    else if(layer.name == "activation" || layer.name == "scale_activation") continue;
                                                    net.layers.push_back(t_layer);
                                                }
                                            }
                                            else if(in_block.name == "3"){
                                                if(MBConv.name == "0")   t_layer.from_idx = {-1};
                                                else    t_layer.from_idx = {-1, -6};
                                                t_layer.name = "ConvBn";
                                                net.layers.push_back(t_layer);
                                            }
                                        }
                                        if(MBConv.name != "0"){
                                            t_layer.layer = residual;
                                            t_layer.name = "Residual";
                                            t_layer.from_idx = {CURRENT_LAYERS, PREV_LAYERS};
                                            net.layers.push_back(t_layer);
                                        }
                                    }
                                }else if(block.name == "stochastic_depth")   continue;
                            }
                        }
                    }
                }
            }
            else{   //children.name: "classifier"
                for(auto child : children.value.named_children()){
                    if(child.name == "0")   continue;   //dropout
                    t_layer.layer = child.value;
                    t_layer.from_idx = {-1};
	    			t_layer.name = "linear";
	    			net.layers.push_back(t_layer); 
                }
            }
        }
    }
}

void *predict_efficientnet(Net *efficientnet){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
		int i;
		float time;
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
		for(i=0;i<efficientnet->layers.size();i++){
			pthread_mutex_lock(&mutex_t[efficientnet->index_n]);
			cond_i[efficientnet->index_n] = 1;
			
			netlayer nl;
			nl.net = efficientnet;
			nl.net->index = i;

			th_arg th;
			th.arg = &nl;

			thpool_add_work(thpool,(void(*)(void *))forward_efficientnet,&th);
			
            while (cond_i[efficientnet->index_n] == 1)
			{
				pthread_cond_wait(&cond_t[efficientnet->index_n], &mutex_t[efficientnet->index_n]);
			}
			i = efficientnet->index;
			efficientnet->input.clear();
			efficientnet->input.push_back(efficientnet->layers[i].output);
			pthread_mutex_unlock(&mutex_t[efficientnet->index_n]);
            cudaStreamSynchronize(streams[efficientnet->H_L][(efficientnet->index_s)%n_streamPerPool]);
		}

		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time, start, end);
        
		std::cout << "\n*****"<<efficientnet->name<<" result " <<time/1000<<"s ***** \n";
		std::cout << (efficientnet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void forward_efficientnet(th_arg *th){
    	{
		at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]);
		pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
		netlayer *nl = th->arg;
		std::vector<torch::jit::IValue> inputs = nl->net->input;
		int k = nl->net->index;

		char str[30];
		sprintf(str, "Efficient layer - %d", k);
		nvtxRangeId_t id1 = nvtxRangeStartA(str);
		
        at::Tensor out;
		cudaEvent_t start, end;
		float l_time;
		// cudaEventCreate(&start);
		// cudaEventCreate(&end);
		// cudaEventRecord(start);

		if(nl->net->layers[k].name == "avgpool"){
            nl->net->identity = inputs[0].toTensor();
            // out = nl->net->layers[k].layer.forward(inputs).toTensor();

            // if(k+1 < nl->net->layers.size()){
            //     nl->net->layers[k].output = out;
            //     k++;
            //     inputs.clear();
            //     inputs.push_back(out);
            // }
        }
        if(nl->net->layers[k].name == "Residual"){
            int add_index = k + nl->net->layers[k].from_idx[0];
			out = nl->net->layers[add_index].output;
			for(int i=1;i<nl->net->layers[k].from_idx.size();i++){
				int add_index = k + nl->net->layers[k].from_idx[i];
				out += nl->net->layers[add_index].output;
			}
        }
        else if(k==nl->net->flatten){	//flatten
            out = inputs[0].toTensor().view({nl->net->layers[k-1].output.size(0), -1});
            inputs.clear();
            inputs.push_back(out);
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
		}
        else{
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
            if(nl->net->layers[k].name == "ConvSiLU"){
                out = F::silu(out);
            }
            else if(nl->net->layers[k].name == "ConvSigmoid"){
                out = torch::sigmoid(out);
                out = nl->net->identity * out;
            }
        }

		cudaStreamSynchronize(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]);

		nl->net->layers[k].output = out;
		nl->net->index = k;
		cond_i[nl->net->index_n]=0;

		nvtxRangeEnd(id1);
		// cudaEventRecord(end);
		// cudaEventSynchronize(end);
		// cudaEventElapsedTime(&l_time, start, end);
		//fprintf((nl->net->fp),"%d,%lf\n",nl->net->index,l_time/1000);
		pthread_cond_signal(&cond_t[nl->net->index_n]);
		pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
	}
}