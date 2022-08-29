#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "co_resnet.h"

using namespace std;
namespace F = torch::nn::functional;


void get_submodule_co_resnet(torch::jit::script::Module module, Net &net){
    Layer t_layer;
    for(auto child : module.named_children()){
        if(child.value.children().size() == 0){
            t_layer.layer = child.value;
            if(child.name == "conv1") t_layer.name = "conv";
            else if(child.name == "bn1") t_layer.name = "bn_relu";
            else if(child.name == "relu") continue;
            else if(child.name == "maxpool") t_layer.name = "maxpool";
            else if(child.name == "avgpool") t_layer.name = "avgpool";
            else if(child.name == "fc") t_layer.name = "fc";
            net.layers.push_back(t_layer);
        }else{  //child.name == layer1~
            for(auto block : child.value.named_children()){ //block.name == 0~ (Bottleneck)
                for(auto layer : block.value.named_children()){ // in Bottleneck block
                    t_layer.layer = layer.value;
                    if(layer.name == "conv1") t_layer.name = "conv1";
                    else if(layer.name == "conv2") t_layer.name = "conv2";
                    else if(layer.name == "conv3") t_layer.name = "conv3";
                    else if(layer.name == "relu") continue;
                    else if(layer.name == "bn1" || layer.name == "bn2") t_layer.name = "bn_relu";
                    else if(layer.name == "bn3") t_layer.name = "bn3";
                    else if(layer.name == "downsample") t_layer.name = "downsample";
                    net.layers.push_back(t_layer);
                }
            }
        }
    }
}

void *predict_co_resnet(Net *res){
    {
        at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});

        int i;
        int j;
		float time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
		std::vector<torch::jit::IValue> origin_input = res->input; /* for several inputs */
        cudaEventRecord(start);
		for(j = 0;j < total_dnn_iter; j++){	/* for several inputs */
			res->input = origin_input;
        for(i = 0;i<res->layers.size();i++) {
            pthread_mutex_lock(&mutex_t[res->index_n]);
            cond_i[res->index_n] = 1; 
            
            netlayer nl;
            nl.net = res;
            nl.net->index = i;

            th_arg th;
            th.arg = &nl;
            thpool_add_work(thpool,(void(*)(void *))forward_resnet,&th);

            while (cond_i[res->index_n] == 1)
            {
                pthread_cond_wait(&cond_t[res->index_n], &mutex_t[res->index_n]);
            }
            i = nl.net->index;
            res->input.clear();
            res->input.push_back(res->layers[i].output);
            pthread_mutex_unlock(&mutex_t[res->index_n]);

            /* FIFOQ 수정 -1 */
            cudaStreamSynchronize(streams[res->H_L][(res->index_s)%n_streamPerPool]);
        }
		}

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        //double time2 = what_time_is_it_now();
        std::cout << "\n*****"<<res->name<<" result  "<<time/1000<<"s ***** \n";
        std::cout << "index num = "<< res->index_n << "	priority num = "<< res->priority << std::endl;
        std::cout << "Stream [" << res->H_L << "][" << (res->index_s)%n_streamPerPool <<"]" << std::endl;
        // std::cout << (res->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
    }
}

void *predict_warm_co_resnet(Net *res){
    {
        at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});

        int i;
        float time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        for(i = 0;i<res->layers.size();i++) {
            pthread_mutex_lock(&mutex_t[res->index_n]);
            cond_i[res->index_n] = 1; 
            
            netlayer nl;
            nl.net = res;
            nl.net->index = i;

            th_arg th;
            th.arg = &nl;
            thpool_add_work(thpool,(void(*)(void *))forward_resnet,&th);

            while (cond_i[res->index_n] == 1)
            {
                pthread_cond_wait(&cond_t[res->index_n], &mutex_t[res->index_n]);
            }
            i = nl.net->index;
            res->input.clear();
            res->input.push_back(res->layers[i].output);
            pthread_mutex_unlock(&mutex_t[res->index_n]);
            /* FIFOQ 수정 -1 */
            cudaStreamSynchronize(streams[res->H_L][(res->index_s)%n_streamPerPool]);
        }

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        //double time2 = what_time_is_it_now();
        std::cout << "\n*****"<<res->name<<" result  "<<time/1000<<"s ***** \n";
        std::cout << "index num = "<< res->index_n << "	priority num = "<< res->priority << std::endl;
        std::cout << "Stream [" << res->H_L << "][" << (res->index_s)%n_streamPerPool <<"]" << std::endl;
        // std::cout << (res->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
    }
}

void forward_co_resnet(th_arg *th){
    {
/* FIFOQ 수정 -2 */
        at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][(th->arg->net->index_s)%n_streamPerPool]); // high, low
        pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
        netlayer *nl = th->arg;
        std::vector<torch::jit::IValue> inputs = nl->net->input; 
        at::Tensor identity = nl->net->identity;
        vector<torch::jit::IValue> inputs_cpy;
        int k =nl->net->index;
        // char str[30];
        // sprintf(str, "Resnet layer - %d", k);
        // nvtxRangeId_t id1 = nvtxRangeStartA(str);
        at::Tensor out;
        cudaEvent_t start, end;
        float l_time;
        // cudaEventCreate(&start);
        // cudaEventCreate(&end);
        // cudaEventRecord(start);
        if(nl->net->layers[k].name == "conv1"){ 
            identity = inputs[0].toTensor(); 
        }
        if(k == nl->net->flatten) //flatten
        {	 
            out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
            inputs.clear();
            inputs.push_back(out);
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
        } 
        else if(nl->net->layers[k].name == "conv3"){ 
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
            if(k+1 < nl->net->layers.size() && nl->net->layers[k+1].name == "bn3"){
				nl->net->layers[k].output = out;
				k++;
				inputs.clear();
                inputs.push_back(out);
				out = nl->net->layers[k].layer.forward(inputs).toTensor();
			}
            if(nl->net->layers[k+1].name != "downsample" ){
                out += identity;
				out = torch::relu(out);
            }
        }
        else if(nl->net->layers[k].name == "downsample"){   // downsample
                inputs_cpy.clear();
                inputs_cpy.push_back(identity); 
                identity = nl->net->layers[k].layer.forward(inputs_cpy).toTensor();
                out = nl->net->layers[k-1].output;
                out += identity;
                out = torch::relu(out);
            }         
        else{ 
            out = nl->net->layers[k].layer.forward(inputs).toTensor(); 
            if(nl->net->layers[k+1].name=="bn_relu"){
                nl->net->layers[k].output = out;
				k++;
				inputs.clear();
                inputs.push_back(out);
				out = nl->net->layers[k].layer.forward(inputs).toTensor();  //bn1,2
                out = torch::relu(out); // relu after bn1,2
            }
        }
/* FIFOQ 수정 -3 밑에 아마 필요없을거임 그래서 지움 */
        //cudaStreamSynchronize(streams[th->arg->net->stream_id[0]]); // 나중에 지워야함

        nl->net->layers[k].output = out;
        nl->net->identity = identity;
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
