#ifndef RESNET18_H
#define RESNET18_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_resnet(torch::jit::script::Module module, Net &net);
void *predict_resnet(Net *input);
void forward_resnet(th_arg *th);
void *predict_warm_resnet(Net *input);

#endif
