# run in GGN environment 

import glob, os
import numpy as np
import pandas as pd
import pickle
import time

names = list()
graph_losses = list()
times = list()

for adj_path in sorted(glob.glob('output_for_GGN_AIDD/*_adj.pickle')):
    name = adj_path.replace('_adj.pickle', '')
    print('name is: ', name)

    #adj_path = name+'adj_matrix.pickle'
    ts_path = adj_path.replace('_adj', '_data')
    out_path = name+'predicted_matrix.txt'
    time_elapsed = -1
    start = time.time()
    print('python train_bn.py --adj_filepath={} --ts_filepath={} --predicted_matrix_path={} --dyn-type=prob --epoch_num=40 --experiments=1'.format(adj_path, ts_path, out_path))
    os.system('python train_bn.py --adj_filepath={} --ts_filepath={} --predicted_matrix_path={} --dyn-type=prob --epoch_num=40'.format(adj_path, ts_path, out_path))
    end = time.time()
    time_elapsed = end - start
    time.sleep(1.0)
    times.append(time_elapsed)


    graph_loss = -1.0
    try:
        adj_gt = pickle.load(open(adj_path, "rb"))
        adj_pred = np.loadtxt(out_path+'.final.txt')
        np.fill_diagonal(adj_pred, 0)
        adj_pred = (adj_pred + np.transpose(adj_pred))/2.0
        diff_matrix = np.abs(adj_gt-adj_pred)
        graph_loss = np.sum(diff_matrix)/2.0
    except:
        pass

    print(graph_loss)
    names.append(name)
    graph_losses.append(graph_loss)
    df = pd.DataFrame({'graph_loss':graph_losses, 'name': names, 'time': times})
    df.to_csv('exp_summary.csv')
