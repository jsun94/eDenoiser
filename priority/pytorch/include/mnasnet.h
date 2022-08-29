#ifndef MNAS_H
#define MNAS_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_MNASNet(torch::jit::script::Module module, Net &child);
void *predict_MNASNet(Net *input);
void forward_MNASNet(th_arg *th);
void *predict_warm_MNASNet(Net *input);

//for residual
#define CURRENT_LAYERS -1   //result of the current layers
#define PREV_LAYERS -7      //result of the previous layers

#endif