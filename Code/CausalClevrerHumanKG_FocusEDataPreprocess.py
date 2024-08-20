import pandas as pd
import numpy as np
import dask.dataframe as dd 
from ampligraph.evaluation import train_test_split_no_unseen

def trainTestValidSplit(folder, filterPredicates):
    # Read the CausalClevrerHumansKG FocusE 
    tceKG=pd.read_csv(r"./Data/CausalCLEVRERHumanKG_AvgWeight.txt",names=['h','r','t','tce'],header=None, sep=",")

    # filter the predicates 
    tceKG = tceKG.loc[tceKG['r'].isin(filterPredicates)]

    data = tceKG.to_numpy()

    # data
    X_train_valid, X_test = train_test_split_no_unseen(data, test_size=0.1)
    X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=0.1)

    X_train_df = pd.DataFrame(X_train, index=None, columns=None)
    X_test_df = pd.DataFrame(X_test, index=None, columns=None)
    X_valid_df = pd.DataFrame(X_valid, index=None, columns=None)

    X_train_df.to_csv('./Data/CausalCLEVERERHumanKG_RandomSplit/'+folder[0]+'/train.txt',columns=None, header=False, index=False)
    X_test_df.to_csv('./Data/CausalCLEVERERHumanKG_RandomSplit/'+folder[0]+'/test.txt',columns=None, header=False, index=False)
    X_valid_df.to_csv('./Data/CausalCLEVERERHumanKG_RandomSplit/'+folder[0]+'/valid.txt',columns=None, header=False, index=False) 
    return


if __name__ == "__main__":
    
    filterTestFlag=[
        # Causal explanation 
        {'CausedByTypeCausesCausedBy':['http://semantic.bosch.com/causal/v00/causedByType','http://semantic.bosch.com/causal/v00/causes','http://semantic.bosch.com/causal/v00/causedBy']},             
        # Causal prediction
         {'CausesType':['http://semantic.bosch.com/causal/v00/causesType']},               
                   ]
    
    count = 1
    for i in filterTestFlag:
        print("Count:",count)
        print("Test KG contains predicates:",i)
        trainTestValidSplit(list(i.keys()), list(i.values())[0])
        count=count+1
        print("\n")
