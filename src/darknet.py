#non so perchè si chiami darknet.py, il sito diceva così e io mi fido di lui

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

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
                pad = (kernel_size - 1)
            else:
                pad = 0

            #convolutional
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #batch normalize
            if batch_normalize:
                bn = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
                module.add_module("batch_norm{0}".format(index), bn)

            #activation type (linear o leakyReLu)
            if activation == "leaky":
                activ = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activ)

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
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

        module_list.append(module) #alla fine del ciclo aggiungo il modulo appena creato
        prev_filters = filters
        output_filters.append(filters)

    return(net_info, module_list)
    
blocks = parse_cfg("cfg/yoloNetwork.cfg") #ottengo e sainifico i dati
print(create_modules(blocks)) #dai dati costruisco il modello di una rete neurale
            
