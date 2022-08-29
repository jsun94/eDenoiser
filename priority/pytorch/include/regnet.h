#ifndef REGNET_H
#define REGNET_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_regnet(torch::jit::script::Module module, Net &child);
void *predict_regnet(Net *input);
void forward_regnet(th_arg *th);
void *predict_warm_regnet(Net *input);

//for residual
#define CURRENT_LAYERS -1   //result of the current layers
#define REG_PREV_LAYERS -7    //result of the previous layers

#endif

