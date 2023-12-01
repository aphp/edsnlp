import torch
import sys
import os


model = torch.load(sys.argv[1], map_location=torch.device('cpu'))
bert_model = model.bert
torch.save(bert_model, sys.argv[2])
