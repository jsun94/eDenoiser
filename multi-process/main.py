import torch
import os
import sys


n_origin_denoiser = int(sys.argv[1])
n_decomposed_denoiser = int(sys.argv[2])
n_inception = int(sys.argv[3])
n_de_inception = int(sys.argv[4])
n_reg = int(sys.argv[5])
n_de_reg = int(sys.argv[6])
n_res = int(sys.argv[7])
n_de_res = int(sys.argv[8])
n_vgg = int(sys.argv[9])
n_de_vgg = int(sys.argv[10])
n_co_res = int(sys.argv[11])
n_co_reg = int(sys.argv[12])
n_resX = int(sys.argv[13])
n_wide = int(sys.argv[14])

n_iter = 1
n_dnn_iter = 1


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


torch.cuda.set_device(0)
device = torch.device('cuda:0')

def inference(my_model, my_example):
    out = my_model(my_example)
    print(f"{my_model.original_name}: {out.shape}")
	# my_model(my_example)
	# print(my_model.original_name, ' done')
    return my_model.original_name

def inference_several(my_model, my_example):
	for iter in range(n_iter):
		out = my_model(my_example)
	# print(f"{my_model.original_name}: {out}")
	print(f"{my_model.original_name}")
	return my_model.original_name, n_iter

def inference_dnn_several(my_model, my_example):
	for iter in range(n_dnn_iter):
		out = my_model(my_example)
	# print(f"{my_model.original_name}: {out}")
	print(f"{my_model.original_name}")
	return my_model.original_name, n_dnn_iter

example1 = torch.ones(1, 3 , 224, 224).to(device)
example2 = torch.ones(1, 3 , 299, 299).to(device)

process_list = []
regnet_process_list = []
co_process_list = []


if n_inception:
	print('n_inception is activated')
    # inception = torch.jit.load("/home/kmsjames/very-big-storage/jimin/decomposition/trace_CIFAR_inception_model.pt", map_location=device).eval()
	inception = torch.jit.load("/home/nvidia/joo/HGD_project/multi-process/inception_model.pt", map_location=device).eval()
    
	for i in range(4):    # warming
		inception(example2)
	for i in range(n_inception):
		# inception_process_list.append(inception)
		process_list.append(inception)

if n_de_inception:
	print('n_de_inception is activated')
    # inception = torch.jit.load("/home/kmsjames/very-big-storage/jimin/decomposition/trace_CIFAR_inception_model.pt", map_location=device).eval()
	de_inception = torch.jit.load("/home/nvidia/joo/HGD_project/trace_imagenet_decomposed_inception_model_train.pt", map_location=device).eval()
    
	for i in range(4):    # warming
		de_inception(example2)
	for i in range(n_de_inception):
		# inception_process_list.append(inception)
		process_list.append(de_inception)

if n_origin_denoiser:
	print('n_origin is activated')
	origin_denoiser = torch.jit.load('/home/nvidia/joo/HGD_project/traced_origin_denoiser_2.pt', map_location=device).eval()
	for i in range(4):
		origin_denoiser(example2)
	for i in range(n_origin_denoiser):
		# origin_process_list.append(origin_denoiser)
		process_list.append(origin_denoiser)

if n_decomposed_denoiser:
	print('n_decomposed is activated')
	decomposed_denoiser = torch.jit.load('/home/nvidia/joo/HGD_project/traced_4tucker_denoiser.pt', map_location = device).eval()
	for i in range(4):
		decomposed_denoiser(example2)
	for i in range(n_decomposed_denoiser):
		process_list.append(decomposed_denoiser)

if n_reg:
	print('n_reg is activated')
	regnet = torch.jit.load('/home/nvidia/joo/HGD_project/regnet_y_32gf_model.pt', map_location = device).eval()
	for i in range(4):
		regnet(example1)
	for i in range(n_reg):
		regnet_process_list.append(regnet)

if n_de_reg:
	print('n_de_reg is activated')
	de_regnet = torch.jit.load('/home/nvidia/joo/HGD_project/trace_imagenet_regnet_y_32gf_decomposed_model.pt', map_location = device).eval()
	for i in range(4):
		de_regnet(example1)
	for i in range(n_de_reg):
		regnet_process_list.append(de_regnet)

if n_res:
	print('n_res is activated')
	resnet = torch.jit.load('/home/nvidia/joo/HGD_project/resnet_model.pt', map_location = device).eval()
	for i in range(4):
		resnet(example1)
	for i in range(n_res):
		regnet_process_list.append(resnet)

if n_de_res:
	print('n_de_res is activated')
	de_resnet = torch.jit.load('/home/nvidia/joo/HGD_project/trace_imagenet_res152_decomposed_model.pt', map_location = device).eval()
	for i in range(4):
		de_resnet(example1)
	for i in range(n_de_res):
		regnet_process_list.append(de_resnet)

if n_resX:
	print('n_resX is activated')
	resXnet = torch.jit.load('/home/nvidia/joo/HGD_project/resnext_model.pt', map_location = device).eval()
	for i in range(4):
		resXnet(example1)
	for i in range(n_resX):
		co_process_list.append(resXnet)

if n_wide:
	print('n_wide is activated')
	widenet = torch.jit.load('/home/nvidia/joo/HGD_project/wideresnet_model.pt', map_location = device).eval()
	for i in range(4):
		widenet(example1)
	for i in range(n_wide):
		co_process_list.append(widenet)

if n_vgg:
	print('n_vgg is activated')
	vggnet = torch.jit.load('/home/nvidia/joo/HGD_project/vgg_model.pt', map_location = device).eval()
	for i in range(4):
		vggnet(example1)
	for i in range(n_vgg):
		regnet_process_list.append(vggnet)

if n_de_vgg:
	print('n_de_vgg is activated')
	vggnet = torch.jit.load('/home/nvidia/joo/HGD_project/trace_imagenet_VGG_decomposed_model.pt', map_location = device).eval()
	for i in range(4):
		de_vggnet(example1)
	for i in range(n_de_vgg):
		regnet_process_list.append(de_vggnet)

if n_co_reg:
	print('n_co_reg is activated')
	co_regnet = torch.jit.load('/home/nvidia/joo/HGD_project/regnet_y_32gf_model.pt', map_location = device).eval()
	for i in range(4):
		co_regnet(example1)
	for i in range(n_co_reg):
		co_process_list.append(co_regnet)

if n_co_res:
	print('n_co_res is activated')
	co_resnet = torch.jit.load('/home/nvidia/joo/HGD_project/resnet_model.pt', map_location = device).eval()
	for i in range(4):
		co_resnet(example1)
	for i in range(n_co_res):
		co_process_list.append(co_resnet)


print('wait')
while 1:
	if(os.stat("switch_python.txt").st_size !=0):
		break
print('wake up')

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for process in process_list:
	# name = inference(process, example2)
	name, n_process = inference_several(process, example2)
for process in regnet_process_list:
	name, n_process = inference_several(process, example1)
	# name = inference(process, example1)
for process in co_process_list:
	name, n_process = inference_dnn_several(process, example1)
end.record()
torch.cuda.synchronize()

print(f'name: {name}, iter: {n_process}, {start.elapsed_time(end)/1000:.6f}s')

