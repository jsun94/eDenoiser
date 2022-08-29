#ifndef DEVGG_H
#define DEVGG_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_de_vgg(torch::jit::script::Module module,Net &net);
void *predict_de_vgg(Net *input);
void *predict_warm_de_vgg(Net *input);

void forward_de_vgg(th_arg *th);

#endif
