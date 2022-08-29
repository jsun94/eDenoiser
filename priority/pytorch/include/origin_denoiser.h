#ifndef ORDENOISER_H
#define ORDENOISER_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_origin_denoiser(torch::jit::script::Module module, Net &child);
void *predict_origin_denoiser(Net *input);
void forward_origin_denoiser(th_arg *th);
#endif