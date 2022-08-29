#ifndef SQUEEZE_H
#define SQUEEZE_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_squeeze(torch::jit::script::Module module, Net &net);
void *predict_squeeze(Net *input);
void forward_squeeze(th_arg *th);
void *predict_warm_squeeze(Net *input);

#define PREV_IDX_2 -2   //layers before 2
#define PREV_IDX_1 -1   //previous layer

#endif
