#non so perchè si chiami darknet.py, il sito diceva così e io mi fido di lui
from __future__ import division

from torch._C import wait
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def get_test_input():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread("resources/prova.png")
    img = cv2.resize(img, (416, 416))
    img_ = img[:,:,:: - 1].transpose((2,0,1)) # da bgr a rgb
    img_ = img_[np.newaxis,:,:,:] / 255.0 #aggiunge un canale (per batch) e normalizza i valori   
    img_ = torch.from_numpy(img_).float().to(device)
    img_ = Variable(img_)
    return img_

#ottiene dal file di config un array con tutti i blocks (array di array)
def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0] #elimina vuote
    lines = [x for x in lines if x[0] != '#'] #elimina commenti
    lines = [x.rstrip().lstrip() for x in lines] #elimina spazi

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[': #inserisco un nuovo blocco (righa: [nomeBlocco])
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip() #elimine le quadre e ottengo solo il nome
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip() #inserisco nel blocco ad indice key il valore (riga: key=val1,val2,ecc)

    blocks.append(block)
    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

#crea moduli pytorch da un array di blocchi
def create_modules(blocks):
    net_info = blocks[0] #informazioni layer di input e eventuale preprocessing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]): #scarto il primo blocco che è net_info
        module = nn.Sequential()

        #differenzio la costruzione in base al tipo di neurone
        if(x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            #TODO capire cosa sono questi valori
            filters = int(x["filters"])
            padding = int(x["pad"]) #offset da 0 al primo elemento
            kernel_size = int(x["size"]) #numero di thread
            stride = int(x["stride"]) #dimensione di un elemento

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #convolutional
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #batch normalize
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm{0}".format(index), bn)

            #activation type (linear o leakyReLu)
            if activation == "leaky":
                activ = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activ)

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{0}".format(index), upsample)

        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])

            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        elif (x["type"] == "yolo"):
            mask = x["mask"].split(',')
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("detection_{0}".format(index), detection)

        if(torch.cuda.is_available()):
            module = module.cuda()
        module_list.append(module) #alla fine del ciclo aggiungo il modulo appena creato
        prev_filters = filters
        output_filters.append(filters)

    return(net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    #non cambiare la firma, overloading
    #definisce cosa fare tra un nodo e un altro
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {} #tensor usato nel ciclo precedente

        write = 0 #indica che abbiamo trovato il primo oggetto e possiamo quindi concaterare gli altri rivelamenti al primo tensor
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if (module_type == "convolutional" or module_type == "upsample"): #se il modulo richiede un operazione lo eseguo
                x = self.module_list[i](x)
            
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if (len(layers) == 1):
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1] > 0):
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            
            elif (module_type == "shortcut"):
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif (module_type == "yolo"):
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")

        header = np.fromfile(fp, dtype = np.int32, count = 5) #i primi 5 int costituiscono l`header
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32) #il resto sono i pesi

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if (module_type == "convolutional"):
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize: #il layout dei pesi cambia se il nodo ha il flag batch_normalize https://blog.paperspace.com/content/images/2018/04/wts-1.png
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)