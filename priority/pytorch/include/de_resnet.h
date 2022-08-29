#ifndef DERESNET_H
#define DERESNET_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_de_resnet(torch::jit::script::Module module, Net &net);
void *predict_de_resnet(Net *input);
void forward_de_resnet(th_arg *th);

#endif
