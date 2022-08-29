#ifndef DEDENOISER_H
#define DEDENOISER_H

#include "net.h"
#include "test.h"
#include "thpool.h"


void get_submodule_decomposed_denoiser(torch::jit::script::Module module, Net &child);
void *predict_decomposed_denoiser(Net *input);
void forward_decomposed_denoiser(th_arg *th);
#endif