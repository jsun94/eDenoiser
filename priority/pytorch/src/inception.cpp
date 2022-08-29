#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include <thread>
#include <unistd.h>
#include "inception.h"

/*

event_idx : branch_num in inception (for recording event)
input_idx : the index of the input from the current layer
skip : Number of layer modules in one branch (How many more signals do thread have to send)
branch_idx : The last layer index of the branch to determine if the operation is complete(exe_success)

*/

namespace F = torch::nn::functional;
using namespace std;

void get_submodule_inception(torch::jit::script::Module module, Net &net){
    Layer t_layer;    
    Dummy temp;
    for(auto children : module.named_children()){
        if(children.name == "Mixed_5b" || children.name == "Mixed_5c" || children.name == "Mixed_5d"){ //InceptionA
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch_pool"){
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.input_idx = PREV_IDX_7;
                    t_layer.event_idx = ++event_idx;
                    t_layer.name = "avg_pool2d";
                    t_layer.skip = SKIP_IDX_2;
                    net.layers.push_back(t_layer);    
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_7, PREV_IDX_5, PREV_IDX_2, CURRENT_IDX};
                }
                if(branch.name == "branch1x1"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.branch_idx = {NEXT_IDX_2, NEXT_IDX_5, NEXT_IDX_7};
                }
                else if(branch.name == "branch5x5_1"){
                    t_layer.input_idx = PREV_IDX_2;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_2;
                }
                else if(branch.name == "branch5x5_2"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_2, NEXT_IDX_3, NEXT_IDX_5};
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.input_idx = PREV_IDX_4;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_3;
                }
                else if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_5, PREV_IDX_3, NEXT_IDX_2};
                }
                else{
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                }
                t_layer.name = "A_" + branch.name;
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                net.layers.push_back(t_layer);
            }
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.input_idx = CURRENT_IDX;
            t_layer.from_idx = {PREV_IDX_8, PREV_IDX_6, PREV_IDX_3, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "concat";
            t_layer.skip = SKIP_IDX_0;
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_6a"){   //InceptionB
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch3x3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.branch_idx = {NEXT_IDX_3, NEXT_IDX_4};
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.input_idx = PREV_IDX_2;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_3;
                }
                else if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_3, NEXT_IDX_1};
                }
                else{
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "B_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = PREV_IDX_5;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.event_idx = ++event_idx;
                    t_layer.name = "max_pool2d";
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.branch_idx = {PREV_IDX_4, PREV_IDX_1, CURRENT_IDX};
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.input_idx = CURRENT_IDX;
            t_layer.from_idx = {PREV_IDX_5, PREV_IDX_2, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "concat";
            t_layer.skip = SKIP_IDX_0;
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_6b" || children.name == "Mixed_6c" || children.name == "Mixed_6d" || children.name == "Mixed_6e" ){ //InceptionC
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch_pool"){
                    t_layer.input_idx = PREV_IDX_10;
                    t_layer.layer = temp;
                    t_layer.event_idx = ++event_idx;
                    t_layer.exe_success = false;
                    t_layer.name = "avg_pool2d";
                    t_layer.skip = SKIP_IDX_2;
                    net.layers.push_back(t_layer);
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_10, PREV_IDX_7, PREV_IDX_2, CURRENT_IDX};
                }
                else if(branch.name == "branch1x1"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.branch_idx = {NEXT_IDX_3, NEXT_IDX_8, NEXT_IDX_10};
                }
                else if(branch.name == "branch7x7_1"){
                    t_layer.input_idx = PREV_IDX_2;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = SKIP_IDX_3;
                }
                else if(branch.name == "branch7x7_3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_3, NEXT_IDX_5, NEXT_IDX_7};
                }
                else if(branch.name == "branch7x7dbl_1"){
                    t_layer.event_idx = ++event_idx;
                    t_layer.input_idx = PREV_IDX_5;
                    t_layer.skip = SKIP_IDX_5;
                }
                else if(branch.name == "branch7x7dbl_3"){
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.branch_idx = {PREV_IDX_8, PREV_IDX_5, NEXT_IDX_2};
                }
                else{
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.input_idx = CURRENT_IDX;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "C_" + branch.name;
                net.layers.push_back(t_layer);
            }
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.from_idx = {PREV_IDX_11, PREV_IDX_8, PREV_IDX_3, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "concat";
            t_layer.skip = SKIP_IDX_0;
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_7a"){   //InceptionD
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                t_layer.skip = SKIP_IDX_0;
                if(branch.name == "branch7x7x3_1"){
                    t_layer.event_idx = ++event_idx;
                    t_layer.input_idx = PREV_IDX_3;
                    t_layer.skip = SKIP_IDX_4;
                }
                else {
                    t_layer.input_idx = CURRENT_IDX;
                    if(branch.name == "branch3x3_1"){
                        t_layer.skip = SKIP_IDX_2;
                        t_layer.event_idx = ++event_idx;
                    }
                    else if(branch.name == "branch7x7x3_4"){
                        t_layer.branch_idx = {PREV_IDX_4, NEXT_IDX_1};
                    }
                    else if(branch.name == "branch3x3_2"){
                        t_layer.branch_idx = {NEXT_IDX_4, NEXT_IDX_5};
                    }
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "D_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch7x7x3_4"){
                    t_layer.input_idx = PREV_IDX_7;
                    t_layer.layer = temp;
                    t_layer.skip = SKIP_IDX_1;
                    t_layer.event_idx = ++event_idx;
                    t_layer.exe_success = false;
                    t_layer.name = "max_pool2d";
                    t_layer.branch_idx = {PREV_IDX_5, PREV_IDX_1, CURRENT_IDX};
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.from_idx = {PREV_IDX_6, PREV_IDX_2, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.skip = SKIP_IDX_0;
            t_layer.name = "concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_7b" || children.name == "Mixed_7c"){    //InceptionE
            int event_idx = INIT_EVENT_IDX;
            for(auto branch : children.value.named_children()){
                t_layer.skip = SKIP_IDX_0;
                if(branch.name == "branch_pool"){
                    t_layer.input_idx = PREV_IDX_11;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.event_idx = ++event_idx;
                    t_layer.name = "avg_pool2d";
	                t_layer.skip = SKIP_IDX_2;
                    net.layers.push_back(t_layer);
                    t_layer.branch_idx = {PREV_IDX_11, PREV_IDX_7, PREV_IDX_2, CURRENT_IDX}; 
                    t_layer.input_idx = CURRENT_IDX;
                }
                else if(branch.name == "branch3x3_1" || branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
                    t_layer.input_idx = PREV_IDX_2;
                    if(branch.name == "branch3x3_1"){
	                    t_layer.skip = SKIP_IDX_4;
                        t_layer.event_idx = ++event_idx;
                    }
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.event_idx = ++event_idx;
	                t_layer.skip = SKIP_IDX_5;
                    t_layer.input_idx = PREV_IDX_6;
                }
                else{
                    t_layer.input_idx = CURRENT_IDX;
                    if(branch.name == "branch1x1"){
                        t_layer.skip = SKIP_IDX_1;
                        t_layer.event_idx = ++event_idx;
                        t_layer.branch_idx = {NEXT_IDX_4, NEXT_IDX_9, NEXT_IDX_11};
                    }
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "E_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
                    if(branch.name == "branch3x3dbl_3b") t_layer.branch_idx = {PREV_IDX_9, PREV_IDX_5, NEXT_IDX_2}; 
                    else t_layer.branch_idx = {PREV_IDX_4, NEXT_IDX_5, NEXT_IDX_7}; 
                    t_layer.input_idx = CURRENT_IDX;
                    t_layer.from_idx = {PREV_IDX_2, PREV_IDX_1};
                    t_layer.layer = temp;
                    t_layer.skip = SKIP_IDX_0;
                    t_layer.exe_success = false;
                    t_layer.name = "concat";
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.skip = SKIP_IDX_0;
            t_layer.input_idx = CURRENT_IDX;
            t_layer.from_idx = {PREV_IDX_12, PREV_IDX_8, PREV_IDX_3, PREV_IDX_1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.event_idx =INIT_EVENT_IDX;
            t_layer.name = "concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "dropout"){
            continue;
        }
        else if(children.name != "AuxLogits")
        {   
            t_layer.input_idx = CURRENT_IDX;
            t_layer.event_idx = INIT_EVENT_IDX;
            t_layer.layer = children.value;
            t_layer.skip = SKIP_IDX_0;
            t_layer.name = children.name;
            t_layer.exe_success = false;
            net.layers.push_back(t_layer);   
        }
    }
}

void *predict_warm_inception(Net *inception){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
        int i;
        float time;
        cudaEvent_t start, end;
        std::vector<int> stream_id = {(inception->index_s)%n_streamPerPool, abs(inception->index_b)%n_streamPerPool, abs((inception->index_b)-1)%n_streamPerPool, abs((inception->index_b)-2)%n_streamPerPool};
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);

        for(i=0;i<inception->layers.size();i++){
            pthread_mutex_lock(&mutex_t[inception->index_n]);
            cond_i[inception->index_n] = 1;
            inception->layers[i].exe_success = false;

            netlayer nl;
            nl.net = inception;
            nl.net->index = i;

            th_arg th;
            th.arg = &nl;
            // std::cout<<"layer index : "<<i<<" name : "<<nl.net->layers[i].name<<std::endl;
            thpool_add_work(thpool,(void(*)(void *))forward_inception,&th);
            
            while (cond_i[inception->index_n] == 1)
            {
                pthread_cond_wait(&cond_t[inception->index_n], &mutex_t[inception->index_n]);
            }
            i = nl.net->index;
            inception->input.clear();
            inception->input.push_back(inception->layers[i].output);
            pthread_mutex_unlock(&mutex_t[inception->index_n]);
            cudaStreamSynchronize(streams[inception->H_L][stream_id[0]]);
        }
		// cudaStreamSynchronize(streams[inception->H_L][stream_id[0]]);
        // cudaEventRecord(end);
        // cudaEventRecord(end, streams[inception->stream_id[0]]);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        std::cout << "\n*****"<<inception->name<<" result  "<<time/1000<<"s ***** \n";
        std::cout << "index num = "<< inception->index_n << "	priority num = "<< inception->priority << std::endl;
        std::cout << "Stream [" << inception->H_L << "][" << stream_id[0] <<"]" << std::endl;
        // std::cout << (inception->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}


void *predict_inception(Net *inception){
	{
		at::cuda::CUDAGuard guard({at::kCUDA,GPU_NUM});
        int i;
		int j;	/* for several inputs */
        float time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
		std::vector<torch::jit::IValue> origin_input = inception->input; 	/* for several inputs */
/* FIFOQ 코드수정 -1 */
        std::vector<int> stream_id = {(inception->index_s)%n_streamPerPool, abs(inception->index_b)%n_streamPerPool, abs((inception->index_b)-1)%n_streamPerPool, abs((inception->index_b)-2)%n_streamPerPool};

		for(j = 0; j < total_iter; j++){	/* for several inputs */
			inception->input = origin_input;	/* for several inputs */
        for(i=0;i<inception->layers.size();i++){
            pthread_mutex_lock(&mutex_t[inception->index_n]);
            cond_i[inception->index_n] = 1;
            inception->layers[i].exe_success = false;

            netlayer nl;
            nl.net = inception;
            nl.net->index = i;

            th_arg th;
            th.arg = &nl;
            // std::cout<<"layer index : "<<i<<" name : "<<nl.net->layers[i].name<<std::endl;
            thpool_add_work(thpool,(void(*)(void *))forward_inception,&th);
            
            while (cond_i[inception->index_n] == 1)
            {
                pthread_cond_wait(&cond_t[inception->index_n], &mutex_t[inception->index_n]);
            }
            i = nl.net->index;
            inception->input.clear();
            inception->input.push_back(inception->layers[i].output);
            pthread_mutex_unlock(&mutex_t[inception->index_n]);
/* FIFOQ 코드수정 -2 */
            cudaStreamSynchronize(streams[inception->H_L][stream_id[0]]);
        }
		}	/* for several inputs */
/* FIFOQ 코드수정 -3 밑에줄 주석처리*/
		// cudaStreamSynchronize(streams[inception->H_L][stream_id[0]]);
        // cudaStreamSynchronize(streams[inception->stream_id[0]]);
        cudaEventRecord(end);
        // cudaEventRecord(end, streams[inception->stream_id[0]]);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        std::cout << "\n*****"<<inception->name<<" result  "<<time/1000<<"s ***** \n";
        std::cout << "index num = "<< inception->index_n << "	priority num = "<< inception->priority << std::endl;
        std::cout << "Stream [" << inception->H_L << "][" << (stream_id[0])%n_streamPerPool <<"]" << std::endl;
        // std::cout << (inception->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}
}

void forward_inception(th_arg *th){
	{
/* FIFOQ 코드수정 -4 */
        std::vector<int> stream_id = {(th->arg->net->index_s)%n_streamPerPool, abs(th->arg->net->index_b)%n_streamPerPool, abs((th->arg->net->index_b)-1)%n_streamPerPool, abs((th->arg->net->index_b)-2)%n_streamPerPool};
		at::cuda::CUDAStreamGuard guard(streams[th->arg->net->H_L][stream_id[0]]);
        pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
        
        char str[30];
        sprintf(str, "Inception layer - %d", th->arg->net->index);
        nvtxRangeId_t id1 = nvtxRangeStartA(str);

        netlayer *nl = th->arg;
        int k = nl->net->index;
        int n_all = nl->net->n_all;
        std::vector<torch::jit::IValue> inputs;
        // std::cout<<k<<" layer : "<<nl->net->layers[k].name<<std::endl;
        if(nl->net->layers[k].input_idx != 0){
            inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
        }
        else {
            inputs = nl->net->input;
        }
        if(nl->net->layers[k + nl->net->layers[k].skip].skip > 0){ // +1 why? predict for loop 
            nl->net->index = k + nl->net->layers[k].skip - 1;
            cond_i[nl->net->index_n]=0;
            pthread_cond_signal(&cond_t[nl->net->index_n]);
            // std::cout<<nl->net->index<<" skip\n";
        }
        pthread_mutex_unlock(&mutex_t[nl->net->index_n]); 
        at::Tensor out;
        if(nl->net->layers[k].skip > 0){   //branch
            {
/* FIFOQ 코드수정 -5 */
                at::cuda::CUDAStreamGuard guard(streams[nl->net->H_L][stream_id[(nl->net->layers[k].event_idx)%4]]); //event_idx == branch_num
                out = inputs[0].toTensor();
                int T = nl->net->layers[k].skip;
                for(int t=0;t<T;k++,t++){
                    // std::cout<<k<<" layer : "<<nl->net->layers[k].name<<std::endl;

                    if(nl->net->layers[k].input_idx != 0){
                        inputs.clear();
                        inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
                    }
                    else {
                        inputs.clear();
                        inputs.push_back(out);
                    } 
                    
                    if(nl->net->layers[k].name == "concat"){
                        // std::cout<<"***** inner concat : ";

                        std::vector<at::Tensor> cat_input;
                        for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
                            // std::cout<<k + nl->net->layers[k].from_idx[i]<<" \n";
                            cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
                        }
                        out = torch::cat(cat_input, 1);
                    }
                    else if(nl->net->layers[k].name == "avg_pool2d"){
                        out = F::avg_pool2d(out,F::AvgPool2dFuncOptions(3).stride(1).padding(1));
                    }
                    else if(nl->net->layers[k].name == "max_pool2d"){
                        out = F::max_pool2d(out,F::MaxPool2dFuncOptions(3).stride(2));
                    }
                    else{
                        out = nl->net->layers[k].layer.forward(inputs).toTensor();
                    }
                    nl->net->layers[k].output = out;
                }
                k--;
                int record = nl->net->layers[k].event_idx;
/* FIFOQ 코드수정 -6 */
                cudaEventRecord(nl->net->record[record], streams[nl->net->H_L][stream_id[(nl->net->layers[k].event_idx)%4]]);
            }
        }
        else if(nl->net->layers[k].name == "concat"){  //brach out
            std::vector<at::Tensor> cat_input;
            // std::cout<<"***** outer concat : ";
            for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
                // std::cout<<k + nl->net->layers[k].from_idx[i]<<" \n";
                cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
            }
            out = torch::cat(cat_input, 1);
        }
        else if(k == nl->net->flatten){
            out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
            inputs.clear();
            inputs.push_back(out);
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
        }
        else{
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
        }

        if(nl->net->layers[k].event_idx >= 0){
            cudaEventSynchronize(nl->net->record[nl->net->layers[k].event_idx]);
            nl->net->layers[k].output = out;
            nl->net->layers[k].exe_success = true;
        }
        nl->net->layers[k].output = out;

        nvtxRangeEnd(id1);

        pthread_mutex_lock(&mutex_t[nl->net->index_n]);

        if(nl->net->layers[k].exe_success == false){
            cond_i[nl->net->index_n]=0;
            //nl->net->index = k;
            pthread_cond_signal(&cond_t[nl->net->index_n]);
        }
        else{
            for(int i=0;i<nl->net->layers[k].branch_idx.size();i++){
                if(nl->net->layers[k + nl->net->layers[k].branch_idx[i]].exe_success == false){
                    pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
                    return;
                }
            }
            for(int i=0;i<nl->net->layers[k].branch_idx.size();i++){ //complete
                nl->net->layers[k + nl->net->layers[k].branch_idx[i]].exe_success = false;
            }
            nl->net->layers[k].exe_success = false;
            nl->net->index = k + nl->net->layers[k].branch_idx.back(); // last layer index of branch
            cond_i[nl->net->index_n]=0;
            pthread_cond_signal(&cond_t[nl->net->index_n]);
        }
        pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
    }
}

