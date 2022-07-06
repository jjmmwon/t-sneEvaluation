import glob
import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def argparsing():
    parser = argparse.ArgumentParser(description="Rate the PCA")
    parser.add_argument('--file_path', '-f', default='default', help="File path to rate the PCA")

    args = parser.parse_args()

    return args

def main():
    args = argparsing()

    if args.file_path == 'default':
        path = '/home/myeongwon/mw/tsne/result/*/*.json'
        json_files = glob.glob(path)

        for json_file in json_files:
            with open(json_file,'r') as f:
                json_data = json.load(f)

            pca = json_data['pca'][0]
            rand = json_data['random']

            data = [['PCA', pca['Loss'], pca['Steadiness'], pca['Cohesiveness']]]
            rand_data = [ [ f'Random{i}', LSC['Loss'], LSC['Steadiness'], LSC["Cohesiveness"] ] for i, LSC in enumerate(rand, start=1)]

            data = data + rand_data

            save_path = json_file[ : json_file.rfind('.')]
            title = save_path[save_path.rfind('/')+1:]

            df = pd.DataFrame(data, columns=["Init", "Loss", "Steadiness", "Cohesiveness"])
            df.plot(x='Init', y = ["Loss", "Steadiness", "Cohesiveness"], title = title, kind="bar", figsize = (15, 7))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
            plt.savefig(f'{save_path}_barplot.png')
            
            df.plot(x='Init', y = ["Loss"], title = title + '_Loss', kind="bar", figsize = (15,7))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
            plt.savefig(f'{save_path}_barplot_Loss.png')

            df.plot(x='Init', y = ["Steadiness", "Cohesiveness"], title = title + '_SNC', kind="bar", figsize = (15, 7))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
            plt.savefig(f'{save_path}_barplot_SNC.png')


            if 'PCA_Rank' in json_data:
                continue

            l_list = [ pca['Loss'] ]
            s_list = [ pca['Steadiness'] ]
            c_list = [ pca['Cohesiveness'] ]

            for LSC in rand:
                l_list.append(LSC['Loss'])
                s_list.append(LSC['Steadiness'])
                c_list.append(LSC['Cohesiveness'])

            l_list.sort()
            s_list.sort(reverse=True)
            c_list.sort(reverse=True)

            pca_rank = {}
            pca_rank['Loss_Rank'] = l_list.index(pca['Loss'])+1
            pca_rank['Steadiness_Rank'] = s_list.index(pca['Steadiness'])+1
            pca_rank['Cohesiveness_Rank'] = c_list.index(pca['Cohesiveness'])+1

            json_data['PCA_Rank'] = pca_rank

            with open(json_file, 'w') as wf:
                json.dump(json_data, wf, indent="\t")
    else:
        path = args.file_path
        folder_path = path[:path.rfind('/')]

        with open(path,'r') as f:
                json_data = json.load(f)

        if 'PCA_Rank' in json_data:
            print("This file is already done")

        pca = json_data['pca'][0]
        rand = json_data['random']

        l_list = [ pca['Loss'] ]
        s_list = [ pca['Steadiness'] ]
        c_list = [ pca['Cohesiveness'] ]

        for LSC in rand:
            l_list.append(LSC['Loss'])
            s_list.append(LSC['Steadiness'])
            c_list.append(LSC['Cohesiveness'])

        l_list.sort()
        s_list.sort(reverse=True)
        c_list.sort(reverse=True)

        pca_rank = {}
        pca_rank['Loss_Rank'] = l_list.index(pca['Loss'])+1
        pca_rank['Steadiness_Rank'] = s_list.index(pca['Steadiness'])+1
        pca_rank['Cohesiveness_Rank'] = c_list.index(pca['Cohesiveness'])+1

        json_data['PCA_Rank'] = pca_rank

        with open(path, 'w') as wf:
            json.dump(json_data, wf, indent="\t")

if __name__ == '__main__':
    main()
        


