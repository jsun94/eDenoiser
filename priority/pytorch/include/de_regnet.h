#ifndef DEREGNET_H
#define DEREGNET_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_de_regnet(torch::jit::script::Module module, Net &child);
void *predict_de_regnet(Net *input);
void *predict_warm_de_regnet(Net *input);
void forward_de_regnet(th_arg *th);

//for residual
#define CURRENT_LAYERS -1   //result of the current layers
#define REG_PREV_LAYERS -7    //result of the previous layers

#endif

