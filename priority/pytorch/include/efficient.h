#ifndef EFFICIENT_H
#define EFFICIENT_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_efficientnet(torch::jit::script::Module module, Net &child);
void *predict_efficientnet(Net *input);
void forward_efficientnet(th_arg *th);

//for residual
#define CURRENT_LAYERS -1   //result of the current layers
#define PREV_LAYERS_1 -6    //result of the previous layers(for child.name == 1)
#define PREV_LAYERS -7      //result of the previous layers(for child.name == 2~7)

#endif

