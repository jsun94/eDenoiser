#ifndef CUDENOISER_H
#define CUDENOISER_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_custom_denoiser(torch::jit::script::Module module, Net &child);
void *predict_custom_denoiser(Net *input);
void forward_custom_denoiser(th_arg *th);
#endif