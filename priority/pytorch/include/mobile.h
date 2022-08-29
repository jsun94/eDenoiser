#ifndef MOBILE_H
#define MOBILE_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_mobilenet(torch::jit::script::Module module, Net &net);
void *predict_mobilenet(Net *input);
void forward_mobilenet(th_arg *th);
void *predict_warm_mobilenet(Net *input);
#endif

