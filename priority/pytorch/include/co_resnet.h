#ifndef CORESNET18_H
#define CORESNET18_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_co_resnet(torch::jit::script::Module module, Net &net);
void *predict_co_resnet(Net *input);
void forward_co_resnet(th_arg *th);
void *predict_warm_co_resnet(Net *input);

#endif
