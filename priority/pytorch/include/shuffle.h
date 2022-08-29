#ifndef SHUFFLE_H
#define SHUFFLE_H

#include "net.h"
#include "test.h"
#include "thpool.h"

at::Tensor channel_shuffle(at::Tensor x, int groups);
void get_submodule_shuffle(torch::jit::script::Module module, Net &net);
void *predict_shuffle(Net *input);
void forward_shuffle(th_arg *th);
void *predict_warm_shuffle(Net *input);

#define PREV_IDX_1 -1   //previous layer
#define PREV_IDX_2 -2   //layers before 2
#define CHUNK_0 0       //chunk[0] of the chunk result

#endif