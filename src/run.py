import glob
import os
import subprocess

data_path = '/home/myeongwon/mw/tsne/dataset/*'
data_files = glob.glob(data_path)

data_files.remove( data_path[:-1] + 'covtype.csv')
data_files.remove( data_path[:-1] + 'fashion-mnist.csv')
data_files.remove( data_path[:-1] + 'NIPS_1987-2015.csv')
data_files.remove( data_path[:-1] + 'Twitter.csv')
data_files.remove( data_path[:-1] + 'rwm.csv')
data_files.remove( data_path[:-1] + 'Breast_Cancer.csv')
data_files.remove( data_path[:-1] + 'BudgetUK.csv')
data_files.remove( data_path[:-1] + 'Cigar.csv')
data_files.remove( data_path[:-1] + 'dermatology.csv')
data_files.remove( data_path[:-1] + 'HousingData.csv')
data_files.remove( data_path[:-1] + 'energydata_complete.csv')
data_files.remove( data_path[:-1] + 'superconduct.csv')

HPram = {"Perplexity" : [15, 30, 45], "Iteration" : [500, 750, 1000], "Learning_Rate" : [200, 800, -1]}

hpram = [(p, i, lr) for p in HPram['Perplexity'] for i in HPram['Iteration'] for lr in HPram['Learning_Rate'] ]


for file in data_files:
    
    for p, i, lr in hpram:
        cmd = f'python3 TSNE.py -f {file} -P {p} -I {i} -L {lr}'
        print(cmd)
        subprocess.run([cmd], shell=True)



    
