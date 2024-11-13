import fair_kcenter as fk
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--k", type= int, required=True)
    
    args=parser.parse_args()
    df=pd.read_csv(args.input, sep=" ")
    colors=np.array(df.iloc[:, 0])
    X=np.array(df.iloc[:,1:])

    sol=fk.Fairkcenter(args.k)
    sol.fit(X, colors)

    #TODO: store solution
