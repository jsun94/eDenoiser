#ifndef ALEX_H
#define ALEX_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_alexnet(torch::jit::script::Module module, Net &child);
void *predict_alexnet(Net *input);
void forward_alexnet(th_arg *th);
void *predict_warm_alexnet(Net *input);
#endif

