"""Descriptive statistics of label-trained features, not independent label validation."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--results',type=Path,required=True)
    args=a.parse_args();root=args.results;df=pd.read_csv(root/'splits.csv');valid=np.load(root/'valid_mask.npy',allow_pickle=False)
    features=np.load(root/'features_finetuned.npy',allow_pickle=False)[valid]
    info=json.loads((root/'summary.json').read_text('utf-8'));y=df.class_idx.to_numpy()[valid]
    if info['train_schema']=='4class':y=np.array([0,1,0,2,3])[y]
    train=(df['split'].to_numpy()[valid]=='train')
    scaler=StandardScaler().fit(features[train])
    x=scaler.transform(features)
    silhouette=silhouette_samples(x,y,metric='euclidean')
    # Query each stored vector explicitly: discard only its own index, retaining ties.
    neighbours=NearestNeighbors(n_neighbors=2,algorithm='brute').fit(x).kneighbors(x,return_distance=False)
    pred=np.array([y[next(j for j in row if j!=i)] for i,row in enumerate(neighbours)])
    n=min(4000,len(y));indices=np.arange(len(y))
    if n<len(y):indices,_=train_test_split(indices,train_size=n,random_state=42,stratify=y)
    xy=TSNE(n_components=2,perplexity=30,learning_rate='auto',init='pca',random_state=42,max_iter=1000).fit_transform(x[indices])
    pd.DataFrame({'record_index':np.flatnonzero(valid)[indices],'class_index':y[indices],'tsne_x':xy[:,0],'tsne_y':xy[:,1]}).to_csv(root/'tsne_projection.csv',index=False)
    report={'schema':info['train_schema'],'valid_records':int(len(y)),'scaler_fit_subset':'train','scaler_fit_records':int(train.sum()),'silhouette':float(silhouette.mean()),'one_nn_leave_one_record_out':float((pred==y).mean()),
      'classes':info['classes'],'per_class':{k:{'count':int((y==c).sum()),'silhouette':float(silhouette[y==c].mean()),'one_nn':float((pred[y==c]==c).mean())} for c,k in enumerate(info['classes'])},
      'tsne_subsample':n,'tsne_random_state':42,'interpretation':'descriptive label association in a trained representation; no independent-event or label-accuracy guarantee'}
    (root/'feature_statistics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
