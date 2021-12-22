import sys
import json

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

sys.path.append('../')
from tuner.score_function import *
from tuner.utils import make_random_option
from tuner.knobs import generate_config

def set_rf_model():
    th = RandomForestRegressor()
    lat = RandomForestRegressor()
    return [th, lat]


def make_solution_pool(opt, pruned_configs, external_datas, defaults):
    temp_configs = [pd.concat([pruned_configs, external_data], axis = 1) for external_data in external_datas]
    targets = []
    configs = []
    TUNE_EM = ["Totals_Ops/sec","Totals_p99_Latency"]
    for i in range(len(TUNE_EM)):
        temp_configs[i] = temp_configs[i].sort_values(TUNE_EM[i], ascending = [i])

    for i in range(len(TUNE_EM)):
        targets.append(temp_configs[i][[TUNE_EM[i]]].values[0])

    for i in range(len(TUNE_EM)):
        configs.append(temp_configs[i].drop(columns = [TUNE_EM[i]]))

    current_solution_pools = [config[:opt.n_pool].values for config in configs]
    targets = [np.repeat(default, opt.n_pool, axis = 0) for default in defaults]

    return current_solution_pools, targets


def prepare_ATR_learning(opt, top_k_knobs, target_knobs: dict, aggregated_data, target_external_data, index):
    # ag_data = aggregated_data[index]['data']
    # te_data = target_external_data[index]['data']
    columns=['Totals_Ops/sec','Totals_p99_Latency']
    with open("../data/workloads_info.json",'r') as f:
        workload_info = json.load(f)

    workloads=np.array([])
    target_workload = np.array([])
    for workload in range(1,len(workload_info.keys())):
        count = 3000
        if workload != opt.target:
            while count:
                if not len(workloads):
                    workloads = np.array(workload_info[str(workload)])
                    count-=1
                workloads = np.vstack((workloads,np.array(workload_info[str(workload)])))
                count-=1
        else:
            while count:
                if not len(target_workload):
                    target_workload = np.array(workload_info[str(workload)])
                    count-=1
                target_workload = np.vstack((target_workload,np.array(workload_info[str(workload)])))
                count-=1

    top_k_knobs_data = pd.DataFrame(top_k_knobs['data'], columns = top_k_knobs['columnlabels'])
    target_knobs_data = pd.DataFrame(target_knobs['data'], columns = target_knobs['columnlabels'])
    aggregated_data = pd.DataFrame(aggregated_data['data'], columns = [columns[index]])

    workload_infos = pd.DataFrame(workloads, columns = workload_info['info'])
    target_workload = pd.DataFrame(target_workload, columns= workload_info['info'])

    target_external_data = pd.DataFrame(target_external_data['data'], columns = [columns[index]])

    knob_with_workload = pd.concat([top_k_knobs_data, workload_infos],axis=1)
    target_workload = pd.concat([target_knobs_data,target_workload], axis=1)

    #X_train, X_val, y_train, y_val = train_test_split(top_k_knobs, aggregated_data, test_size = 0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(knob_with_workload, aggregated_data, test_size = 0.33, random_state=42)

    scaler_X = StandardScaler().fit(X_train)
    X_tr = scaler_X.transform(X_train).astype(np.float32)
    X_val = scaler_X.transform(X_val).astype(np.float32)
    #X_te = scaler_X.transform(target_knobs).astype(np.float32)

    scaler_y = StandardScaler().fit(y_train)
    y_train = scaler_y.transform(y_train).astype(np.float32)
    y_val = scaler_y.transform(y_val).astype(np.float32)

    return X_tr, y_train

def RF_fitness(solution, model):
    predicts = model.predict(solution)
    return predicts

def ATR_GA(opt, models, targets, top_k_knobs, current_solution_pools, fitness_function, GA_options, scaler_X, scaler_ys, logger):
    n_configs, n_pool_half, mutation = GA_options
    ops_predicts = []
    lat_predicts = []
    from tqdm import tqdm
    for i in tqdm(range(4000)):
        index = i%2 # multi object
        #index = 0 # throughput (single object)
        #index = 1 # latency (single object)
        predicts = []
        for index in range(2):
            scaled_pool = scaler_X.transform(current_solution_pools[index])
            predict = fitness_function(scaled_pool, models[index])
            fitness = scaler_ys[index].inverse_transform(predict)
            predicts.append(fitness)

        #save preidct ops and latency
        if index:
            lat_predicts.append(np.min(predicts))
        else:
            ops_predicts.append(np.max(predicts))
        
        #score function and sort by score
        idx_fitness = ATR_loss2(targets, predicts,[0.5,0.5])
        sorted_idx_fitness = np.argsort(idx_fitness)[n_pool_half:]
        best_solution_pool = current_solution_pools[index][sorted_idx_fitness,:]

        if i % 1000 == 998:
            logger.info(f"[{i+1:3d}/{4000:3d}] best fitness: {max(idx_fitness)}")
        if i % 1000 == 999:
            logger.info(f"[{i+1:3d}/{4000:3d}] best fitness: {max(idx_fitness)}")
        
        #random select crossover ratio
        pivot = np.random.choice(np.arange(1,n_configs))
        new_solution_pool = np.zeros_like(best_solution_pool)
        for j in range(n_pool_half):
            #crossover
            new_solution_pool[j][:pivot] = best_solution_pool[j][:pivot]
            new_solution_pool[j][pivot:n_configs] = best_solution_pool[n_pool_half-1-j][pivot:n_configs]
            new_solution_pool[j][n_configs:] = current_solution_pools[index][0][n_configs:]
            
            #mutation
            import random
            random_knobs = make_random_option(top_k_knobs['columnlabels'])
            knobs_value = list(random_knobs.values())
            random_knob_index = list(range(n_configs))
            random.shuffle(random_knob_index)
            random_knob_index = random_knob_index[:mutation]
            for k in range(len(random_knob_index)):
                new_solution_pool[j][random_knob_index[k]] = knobs_value[random_knob_index[k]]

        current_solution_pools[index] = np.vstack([best_solution_pool, new_solution_pool])

    final_solution_pool = pd.DataFrame(best_solution_pool)
    logger.info(top_k_knobs)
    logger.info(final_solution_pool)
    top_k_config_path, name, connect = generate_config(opt, top_k_knobs['columnlabels'], final_solution_pool)

    return top_k_config_path, name, connect

def server_connection(args, top_k_config_path, name):
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('34.64.145.240', username='jieun', password='1423')

    sftp = client.open_sftp()
    sftp.put(top_k_config_path, './redis-sample-generation/'+name)
    command = f'python ./redis-sample-generation/connection.py {args.persistence.lower()} {args.target} ./redis-sample-generation/'+name
    print("Sftp Start")
    _, ssh_stdout, _ = client.exec_command(command)
    exit_status = ssh_stdout.channel.recv_exit_status()
    if exit_status == 0:
        sftp.get(f'/home/jieun/result_{args.persistence.lower()}_external_GA.csv', f'./GA_config/result_{args.persistence.lower()}_external_GA.csv')
        sftp.get(f'/home/jieun/result_{args.persistence.lower()}_internal_GA.csv', f'./GA_config/result_{args.persistence.lower()}_internal_GA.csv')
    sftp.close()
    client.exec_command('rm ./redis-sample-generation/'+name)
    client.exec_command(f'rm /home/jieun/result_{args.persistence.lower()}_external_GA.csv')
    client.exec_command(f'rm /home/jieun/result_{args.persistence.lower()}_internal_GA.csv')

    client.close()