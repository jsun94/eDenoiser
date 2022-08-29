#!/bin/bash

if [ $# -ne 14 ] ; then
    echo "number of parameter is not 14"
    exit 0
fi

n_origin_denoiser=$1
n_decomposed_denoiser=$2
n_inception=$3
n_de_inception=$4
n_reg=$5
n_de_reg=$6
n_res=$7
n_de_res=$8
n_vgg=$9
n_de_vgg=${10}
n_co_res=${11}
n_co_reg=${12}
n_resX=${13}
n_wide=${14}



for((i=0;i<${n_origin_denoiser};i++));
do
	python main.py 1 0 0 0 0 0 0 0 0 0 0 0 0 0 &
done
for((i=0;i<${n_decomposed_denoiser};i++));
do
	python main.py 0 1 0 0 0 0 0 0 0 0 0 0 0 0 &
done
for((i=0;i<${n_inception};i++));
do
	python main.py 0 0 1 0 0 0 0 0 0 0 0 0 0 0 &
done
for((i=0;i<${n_de_inception};i++));
do
	python main.py 0 0 0 1 0 0 0 0 0 0 0 0 0 0 &
done
for((i=0;i<${n_reg};i++));
do
	python main.py 0 0 0 0 1 0 0 0 0 0 0 0 0 0 &
done
for((i=0;i<${n_de_reg};i++));
do
	python main.py 0 0 0 0 0 1 0 0 0 0 0 0 0 0 &
done
for((i=0;i<${n_res};i++));
do
	python main.py 0 0 0 0 0 0 1 0 0 0 0 0 0 0 &
done
for((i=0;i<${n_de_res};i++));
do
	python main.py 0 0 0 0 0 0 0 1 0 0 0 0 0 0 &
done
for((i=0;i<${n_vgg};i++));
do
	python main.py 0 0 0 0 0 0 0 0 1 0 0 0 0 0 &
done
for((i=0;i<${n_de_vgg};i++));
do
	python main.py 0 0 0 0 0 0 0 0 0 1 0 0 0 0 &
done
for((i=0;i<${n_co_res};i++));
do
	python main.py 0 0 0 0 0 0 0 0 0 0 1 0 0 0 &
done
for((i=0;i<${n_co_reg};i++));
do
	python main.py 0 0 0 0 0 0 0 0 0 0 0 1 0 0 &
done
for((i=0;i<${n_resX};i++));
do
	python main.py 0 0 0 0 0 0 0 0 0 0 0 0 1 0 &
done
for((i=0;i<${n_wide};i++));
do
	python main.py 0 0 0 0 0 0 0 0 0 0 0 0 0 1 &
done