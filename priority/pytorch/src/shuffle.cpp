#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <inttypes.h>
#include <iostream>
#include <string>
#include <memory>
#include "cuda_runtime.h"

#include "shuffle.h"

namespace F = torch::nn::functional;
using namespace std;

void get_submodule_shuffle(torch::jit::script::Module module, Net &net)
{
    Layer t_layer;
    Dummy concat;

    for(auto ch : module.named_children()){ // conv1,5 stage2,3,4 fc, maxpool
            if(ch.name == "stage2" || ch.name == "stage3" || ch.name == "stage4"){
                for(auto seq : ch.value.named_children()){      // seq.name == 0,1,2,3
                        bool exist_branch1 = false;
                        for(auto branch : seq.value.named_children()){  //branch.name == branch1,branch2
                                if(branch.name == "branch1" && (branch.value.children().size() != 0)){  //seq == 0
                                        exist_branch1 = true;
                                        t_layer.input_idx = PREV_IDX_1;
                                        t_layer.name = branch.name;
                                        t_layer.layer = branch.value;
                                        t_layer.exe_success = false;
                                        net.layers.push_back(t_layer);
                                }
                                else if(branch.name == "branch2"){
                                        if(exist_branch1){
                                                t_layer.input_idx = PREV_IDX_2;
                                                t_layer.name = branch.name;
                                        }
                                        else{
                                                t_layer.input_idx = PREV_IDX_1;
                                                t_layer.name = "chunk_and_branch2";
                                        }
                                        t_layer.layer = branch.value;
                                        t_layer.exe_success = false;
                                        net.layers.push_back(t_layer);
                                }
                        }
                        
                        if(exist_branch1){      //seq == 0
                                t_layer.from_idx = {PREV_IDX_2, PREV_IDX_1};
                        }
                        else{
                                t_layer.from_idx = {CHUNK_0, PREV_IDX_1}; 
                        }
                        t_layer.input_idx = PREV_IDX_1;
                        t_layer.name = "concat";
                        t_layer.layer = concat;
                        t_layer.exe_success = false;
                        net.layers.push_back(t_layer);               
                }
            }
            else if(ch.name == "conv1" || ch.name == "conv5"){
                t_layer.input_idx = PREV_IDX_1;
                t_layer.layer = ch.value;
                t_layer.name = "ConvBnRelu";
                t_layer.exe_success = false;
                net.layers.push_back(t_layer);
            }
            else if(ch.name == "fc"){ // fc
                t_layer.input_idx = PREV_IDX_1;
                t_layer.layer = ch.value;
                t_layer.name = "fc";
                t_layer.exe_success = false;
                net.layers.push_back(t_layer);
            }
            else if(ch.name == "maxpool"){
                t_layer.input_idx = PREV_IDX_1;
                t_layer.layer = ch.value;
                t_layer.name = "maxpool";
                t_layer.exe_success = false;
                net.layers.push_back(t_layer);    
            }
        }
}

at::Tensor channel_shuffle(at::Tensor x, int groups){
        int batchsize = x.sizes()[0];
        int num_channels = x.sizes()[1];
        int height = x.sizes()[2];
        int width = x.sizes()[3];
        int channels_per_group = num_channels / groups;
        x = x.view({batchsize,groups, channels_per_group,height, width});
        x = x.transpose(1,2).contiguous();
        x = x.view({batchsize, -1, height, width});

        return x;
}

void *predict_warm_shuffle(Net *shuffle){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
                int i;
                float time;
                cudaEvent_t start, end;
                cudaEventCreate(&start);
                cudaEventCreate(&end);
                cudaEventRecord(start);
/* FIFOQ 수정 -1 */
                std::vector<int> stream_id = {(shuffle->index_s)%n_streamPerPool, abs(shuffle->index_b)%n_streamPerPool};
                for(i=0;i<shuffle->layers.size();i++){
                        pthread_mutex_lock(&mutex_t[shuffle->index_n]);
                        cond_i[shuffle->index_n] = 1;
                        shuffle->layers[i].exe_success = false;

                        netlayer nl;
                        nl.net = shuffle;
                        nl.net->index = i;

                        th_arg th;
                        th.arg = &nl;

                        thpool_add_work(thpool,(void(*)(void *))forward_shuffle,&th);

                        while (cond_i[shuffle->index_n] == 1)
                        {
                                pthread_cond_wait(&cond_t[shuffle->index_n], &mutex_t[shuffle->index_n]);
                        }
                        shuffle->input.clear();
                        shuffle->input.push_back(shuffle->layers[i].output);
                        pthread_mutex_unlock(&mutex_t[shuffle->index_n]);
/* FIFOQ 수정 -2 */
                        cudaStreamSynchronize(streams[shuffle->H_L][stream_id[0]]);
                }
/* FIFOQ 수정 -3 */
                //cudaStreamSynchronize(streams[shuffle->stream_id[0]]);
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&time, start, end);
                std::cout << "\n*****"<<shuffle->name<<" result  "<<time/1000<<"s ***** \n";
                std::cout << "index num = "<< shuffle->index_n << "	priority num = "<< shuffle->priority << std::endl;
                std::cout << "Stream [" << shuffle->H_L << "][" << stream_id[0] << "]" << std::endl;
                // std::cout<<(shuffle->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) <<"\n";
        }
}


void *predict_shuffle(Net *shuffle){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
                int i;
				int j;
				std::vector<torch::jit::IValue> origin_input = shuffle->input;
                float time;
                cudaEvent_t start, end;
                cudaEventCreate(&start);
                cudaEventCreate(&end);
                cudaEventRecord(start);
/* FIFOQ 수정 -1 */
                std::vector<int> stream_id = {(shuffle->index_s)%n_streamPerPool, abs(shuffle->index_b)%n_streamPerPool};
                for(j=0;j<total_dnn_iter;j++){
					shuffle->input = origin_input;
				for(i=0;i<shuffle->layers.size();i++){
                        pthread_mutex_lock(&mutex_t[shuffle->index_n]);
                        cond_i[shuffle->index_n] = 1;
                        shuffle->layers[i].exe_success = false;

                        netlayer nl;
                        nl.net = shuffle;
                        nl.net->index = i;

                        th_arg th;
                        th.arg = &nl;

                        thpool_add_work(thpool,(void(*)(void *))forward_shuffle,&th);

                        while (cond_i[shuffle->index_n] == 1)
                        {
                                pthread_cond_wait(&cond_t[shuffle->index_n], &mutex_t[shuffle->index_n]);
                        }
                        shuffle->input.clear();
                        shuffle->input.push_back(shuffle->layers[i].output);
                        pthread_mutex_unlock(&mutex_t[shuffle->index_n]);
/* FIFOQ 수정 -2 */
                        cudaStreamSynchronize(streams[shuffle->H_L][stream_id[0]]);
                }
				}
/* FIFOQ 수정 -3 */
                //cudaStreamSynchronize(streams[shuffle->stream_id[0]]);
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&time, start, end);
                std::cout << "\n*****"<<shuffle->name<<" result  "<<time/1000<<"s ***** \n";
                std::cout << "index num = "<< shuffle->index_n << "	priority num = "<< shuffle->priority << std::endl;
                std::cout << "Stream [" << shuffle->H_L << "][" << stream_id[0] << "]" << std::endl;
                // std::cout<<(shuffle->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) <<"\n";
        }
}

void forward_shuffle(th_arg *th){
	{
/* FIFOQ 수정 -4 */
                std::vector<int> stream_id = {(th->arg->net->index_s)%n_streamPerPool, abs(th->arg->net->index_b)%n_streamPerPool};
/* FIFOQ 수정 -5 */
                at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][stream_id[0]]); // high, low
                pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
                netlayer *nl = th->arg;
                std::vector<torch::jit::IValue> inputs;
                int k = nl->net->index;
                // int n_all = nl->net->n_all;
                int j;
                //at::cuda::setCurrentCUDAStream(streams[(nl->net->index_n)]);


                if(k==0) 
                        inputs = nl->net->input;
                else{
                        inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
                }
                if(nl->net->layers[k].name == "branch1"){
                        cond_i[nl->net->index_n]=0;
                        pthread_cond_signal(&cond_t[nl->net->index_n]);
                }
                pthread_mutex_unlock(&mutex_t[nl->net->index_n]);

                at::Tensor out;

                if(k == nl->net->gap){  //mean
                        out = inputs[0].toTensor().mean({2,3});
                        inputs.clear();
                        inputs.push_back(out);
                        out = nl->net->layers[k].layer.forward(inputs).toTensor();
                }
                else if(nl->net->layers[k].name == "concat"){
                        std::vector<at::Tensor> cat_input;
                        for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
                                if(nl->net->layers[k].from_idx[i]>=0){
                                        cat_input.push_back(nl->net->chunk[nl->net->layers[k].from_idx[i]]);
                                }
                                        
                                else{
                                        cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
                                }
                                        
                        }
                        out = torch::cat(cat_input, 1);
                        out = channel_shuffle(out, 2);
                }else{
                        //chunk_and_branch , branch1, branch2 , conv, maxpool
                        if(nl->net->layers[k].name == "branch1"){
                                {
/* FIFOQ 수정 -6 */
                                        at::cuda::CUDAStreamGuard guard(streams[nl->net->H_L][stream_id[0]]); // stream_id[0] = nl->net->index
                                        j=1;
                                        out = nl->net->layers[k].layer.forward(inputs).toTensor();
/* FIFOQ 수정 -7 */
                                        cudaEventRecord(nl->net->record[1],streams[nl->net->H_L][stream_id[0]]);
                                }
                        }else if(nl->net->layers[k].name == "branch2"){
                                {
/* FIFOQ 수정 -8 */
                                        at::cuda::CUDAStreamGuard guard(streams[nl->net->H_L][stream_id[1]]); // stream_id[1] = 29 (32-3)
                                        j=-1;
                                        out = nl->net->layers[k].layer.forward(inputs).toTensor();
/* FIFOQ 수정 -9 */
                                        cudaEventRecord(nl->net->record[1],streams[nl->net->H_L][stream_id[1]]);
                                }
                        }else if(nl->net->layers[k].name == "chunk_and_branch2"){
                                nl->net->chunk = inputs[0].toTensor().chunk(2,1);
                                inputs.clear();
                                inputs.push_back(nl->net->chunk[1]);
                                out = nl->net->layers[k].layer.forward(inputs).toTensor();
                        }else{
                                
                                out = nl->net->layers[k].layer.forward(inputs).toTensor();
                        }
                }
       
                if(nl->net->layers[k].name == "branch1"){
                        cudaEventSynchronize(nl->net->record[0]);
                        nl->net->layers[k].output = out;
                        nl->net->layers[k].exe_success = true;
                }
                else if(nl->net->layers[k].name == "branch2"){
                        cudaEventSynchronize(nl->net->record[1]);
                        nl->net->layers[k].output = out;
                        nl->net->layers[k].exe_success = true;
                }
                else
                        nl->net->layers[k].output = out;

                pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
                if(nl->net->layers[k].name != "branch1" && nl->net->layers[k].name != "branch2"){
                        cond_i[nl->net->index_n]=0;
                        pthread_cond_signal(&cond_t[nl->net->index_n]);
                }else if(nl->net->layers[k].exe_success && nl->net->layers[k+j].exe_success){
                        nl->net->layers[k].exe_success = false;
                        nl->net->layers[k+j].exe_success = false;
                        cond_i[nl->net->index_n]=0;
                        pthread_cond_signal(&cond_t[nl->net->index_n]);
                }
                pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
        }
}