# run in GGN environment 

import glob, os
import numpy as np
import pandas as pd
import pickle
import time
import torch

names = list()
graph_losses = list()

for adj_path in sorted(glob.glob('data/*_adj.pickle')):
    data_path = adj_path.replace('_adj', '_data')
    print('name is: ', adj_path)
    #gol_GRIDstep1_25_id0_adj.pickle
    dynname = adj_path.split('/')[-1].split('_')[-5]
    networkname = adj_path.split('_')[-4]
    nodenum = adj_path.split('_')[-3]

    command = 'python train_ginabaseline.py --device_id=0 --network={} --nodes={} --sys={}'.format(networkname, nodenum, dynname)

    time_elapsed = -1
    start = time.time()
    print(command)
    os.system(command)
    end = time.time()
    time_elapsed = end - start
    time.sleep(1.0)

    graph_loss = -1.0

    try:
        adj_gt = pickle.load(open(adj_path, "rb"))
        adj_out_name = 'model/adj_{}_{}_25_id1.pkl'.format(dynname, networkname)
        adj_pred = torch.load(adj_out_name).detach().cpu().numpy()
        np.fill_diagonal(adj_pred, 0)
        adj_pred = (adj_pred + np.transpose(adj_pred))/2.0
        adj_pred = np.where(adj_pred>0.5, 1.0, 0.0)
        diff_matrix = np.abs(adj_gt-adj_pred)
        graph_loss = np.sum(diff_matrix)/2.0  
    except:
        pass

    print(graph_loss)
    names.append(adj_path)
    graph_losses.append(graph_loss)
    df = pd.DataFrame({'graph_loss':graph_losses, 'name': names, 'time': time_elapsed})
    df.to_csv('exp_summary.csv')
    print(df)