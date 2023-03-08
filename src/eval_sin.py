
import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler
import matplotlib.pyplot as plt

def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


n_points = 11
max_val = 4* np.pi 
min_val = 0
b_size = 1
a_b = 1.2 #1 to 2
b_b = 0.5*np.pi #0 to 2*np.pi


model, conf = get_model_from_run("/home/katie/Desktop/in-context-learning/models/sin_regression/bd87e335-0324-41a6-9676-d2d216f6cdd1")

if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
    device = "cuda"
else:
    device = "cpu"

model.to(device)

xs_b_p_all = torch.arange(start=0, end=4*np.pi, step=4*np.pi/100)
real_y = []
pred_y = []

for j in range(100):
	xs_b = torch.arange(start=0,end=4*np.pi, step=4*np.pi/10).unsqueeze(0).unsqueeze(-1) #torch.rand(b_size, n_points, 1)*(2*np.pi)
	# xs_b = torch.rand(b_size, 10, 1)*(4*np.pi)
	xs_b_p = xs_b_p_all[j].reshape((1,1,1))#torch.rand(b_size, n_points, 1)*(4*np.pi-2*np.pi)+1*np.pi

	i = 10
	xs_comb = torch.cat((xs_b, xs_b_p), dim=1)
	ys = torch.sin(a_b*xs_comb + b_b)

	print(ys[:, 10])
	real_y.append(ys[:, 10][0][0].item())

	pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()

	print(pred)
	pred_y.append(pred[0][0].item())

plt.plot(xs_b_p_all.detach().cpu().numpy(), real_y)
plt.plot(xs_b_p_all.detach().cpu().numpy(), pred_y)
plt.scatter(xs_b.detach().cpu().numpy(), ys[:, :-1].detach().cpu().numpy())
plt.show()
