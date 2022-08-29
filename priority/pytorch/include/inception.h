#ifndef INCEPTION_H
#define INCEPTION_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_inception(torch::jit::script::Module module, Net &child);
void *predict_inception(Net *input);
void forward_inception(th_arg *th);

/*
PREV_IDX_num: layers before num
CURRENT_IDX: current layer
NEXT_IDX_num: layers after num
INIT_EVENT_IDX: initialization value(event_idx == -1: branch X, evnet_idx>=0: branch)
num in SKIP_IDX_num: Number of layer modules in one branch
*/
#define PREV_IDX_12 -12
#define PREV_IDX_11 -11
#define PREV_IDX_10 -10
#define PREV_IDX_9 -9
#define PREV_IDX_8 -8
#define PREV_IDX_7 -7
#define PREV_IDX_6 -6
#define PREV_IDX_5 -5
#define PREV_IDX_4 -4
#define PREV_IDX_3 -3
#define PREV_IDX_2 -2
#define PREV_IDX_1 -1
#define CURRENT_IDX 0       //current layer
#define NEXT_IDX_1 1        
#define NEXT_IDX_2 2
#define NEXT_IDX_3 3
#define NEXT_IDX_4 4
#define NEXT_IDX_5 5
#define NEXT_IDX_7 7
#define NEXT_IDX_8 8
#define NEXT_IDX_9 9
#define NEXT_IDX_10 10
#define NEXT_IDX_11 11
#define INIT_EVENT_IDX -1
#define SKIP_IDX_0 0
#define SKIP_IDX_1 1        
#define SKIP_IDX_2 2
#define SKIP_IDX_3 3
#define SKIP_IDX_4 4
#define SKIP_IDX_5 5

#endif

