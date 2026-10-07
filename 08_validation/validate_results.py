"""Independently verify validation/test scores and reload the saved models."""
from pathlib import Path
import argparse,json,os,sys,hashlib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,confusion_matrix

def scores(y,p,k):
    pred=p.argmax(axis=1);labels=np.arange(k)
    return {'overall_accuracy':accuracy_score(y,pred),
        'macro_precision':precision_score(y,pred,labels=labels,average='macro',zero_division=0),
        'macro_recall':recall_score(y,pred,labels=labels,average='macro',zero_division=0),
        'macro_f1':f1_score(y,pred,labels=labels,average='macro',zero_division=0),
        'balanced_accuracy':balanced_accuracy_score(y,pred)},confusion_matrix(y,pred,labels=labels)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.results
    info=json.loads((root/'summary.json').read_text('utf-8'))
    config=json.loads((root/'run_config.json').read_text('utf-8'))
    signature=json.loads((root/'run_signature.json').read_text('utf-8'))
    assert info['fit_subset']==signature['fit_subset']=='train'
    assert hashlib.sha256(Path(config['split_manifest']).read_bytes()).hexdigest()==info['split_sha256']
    df=pd.read_csv(root/'splits.csv')
    valid=np.load(root/'valid_mask.npy',allow_pickle=False)
    assert valid.all() and len(df)==info['n_samples_total']
    train_idx=np.flatnonzero(df.split.to_numpy()=='train')
    assert len(train_idx)==info['n_train']
    mapping=np.array([0,1,0,2,3]);labels=df.class_idx.to_numpy()
    if info['train_schema']=='4class':labels=mapping[labels]
    features=np.load(root/'features_finetuned.npy',allow_pickle=False)
    members=['CNN','RandomForest'] if info['train_schema']=='4class' else ['CNN','RandomForest','LogisticRegression','MLP','XGBoost']
    assert info['ensemble_members']==members
    models=['CNN','RandomForest','LogisticRegression','MLP','XGBoost','Ensemble']
    report={'schema':info['train_schema'],'fit_subset':'train','phases':{},'reloaded_models':[]}
    for phase,subset,prefix,csvname in [('test','test','','multimodel_metrics.csv'),('validation','val','validation_','validation_metrics.csv')]:
        idx=np.flatnonzero(df.split.to_numpy()==subset)
        y=np.load(root/f'{phase}_labels.npy',allow_pickle=False)
        assert np.array_equal(y,labels[idx])
        records=pd.read_csv(root/f'{phase}_records.csv')
        assert np.array_equal(records.record_index.to_numpy(),idx)
        assert records.relative_path.tolist()==df.iloc[idx].relative_path.tolist()
        stored=pd.read_csv(root/csvname)
        assert stored.model.tolist()==models
        rows=[]
        for row in stored.itertuples():
            p=np.load(root/f'{prefix}proba_{row.model}.npy',allow_pickle=False)
            assert p.shape==(len(y),len(info['classes'])) and np.isfinite(p).all()
            assert np.allclose(p.sum(axis=1),1,atol=2e-6) and p.min()>=0
            result,cm=scores(y,p,len(info['classes']))
            for key,value in result.items():assert abs(value-float(getattr(row,key)))<1e-9,(phase,row.model,key)
            assert np.array_equal(cm,np.load(root/f'{prefix}confusion_{row.model}.npy',allow_pickle=False))
            rows.append({'model':row.model,**result})
        mean=np.mean([np.load(root/f'{prefix}proba_{m}.npy',allow_pickle=False).astype(np.float64) for m in members],axis=0)
        assert np.allclose(mean,np.load(root/f'{prefix}proba_Ensemble.npy',allow_pickle=False),atol=1e-6)
        report['phases'][phase]={'records':len(y),'verified_models':rows}
    for model in models[1:-1]:
        saved=joblib.load(root/f'model_{model}.joblib')
        assert saved['fit_subset']=='train' and np.array_equal(saved['fit_indices'],train_idx)
        assert int(saved['scaler'].n_samples_seen_)==len(train_idx)
        assert np.allclose(saved['scaler'].mean_,features[train_idx].mean(axis=0,dtype=np.float64),atol=1e-7)
        for phase,subset,prefix in [('test','test',''),('validation','val','validation_')]:
            mask=df.split.to_numpy()==subset
            p=saved['classifier'].predict_proba(saved['scaler'].transform(features[mask]))
            assert np.allclose(p,np.load(root/f'{prefix}proba_{model}.npy',allow_pickle=False),atol=1e-7)
        report['reloaded_models'].append(model)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'07_classification'))
    from train_classification import build_resnet18_1ch,predict_cnn_proba
    checkpoint=torch.load(root/'cnn_checkpoint.pt',map_location='cpu',weights_only=False)
    assert checkpoint['fit_subset']=='train' and checkpoint['split_sha256']==info['split_sha256']
    assert checkpoint['fit_indices']==train_idx.tolist()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=build_resnet18_1ch(len(info['classes']),imagenet=False).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    specs=np.load(root/'spectrograms.npy',mmap_mode='r',allow_pickle=False)
    for phase,subset,prefix in [('test','test',''),('validation','val','validation_')]:
        idx=np.flatnonzero(df.split.to_numpy()==subset)
        p=predict_cnn_proba(model,specs,labels,idx,device,config['batch_size'])
        stored=np.load(root/f'{prefix}proba_CNN.npy',allow_pickle=False)
        assert np.allclose(p,stored,atol=2e-6) and np.array_equal(p.argmax(1),stored.argmax(1))
    report['reloaded_models'].append('CNN')
    for tag,filename in [('imagenet_frozen','features_imagenet.npy'),('finetuned','features_finetuned.npy')]:
        saved=joblib.load(root/f'ablation_{tag}.joblib')
        assert saved['fit_subset']=='train' and np.array_equal(saved['fit_indices'],train_idx)
        assert int(saved['scaler'].n_samples_seen_)==len(train_idx)
        feats=np.load(root/filename,allow_pickle=False)
        assert np.allclose(saved['scaler'].mean_,feats[train_idx].mean(axis=0,dtype=np.float64),atol=1e-7)
        for phase,subset in [('test','test'),('validation','val')]:
            mask=df.split.to_numpy()==subset
            p=saved['classifier'].predict_proba(saved['scaler'].transform(feats[mask]))
            assert np.allclose(p,np.load(root/f'ablation_{tag}_{phase}_proba.npy',allow_pickle=False),atol=1e-7)
            expected=json.loads((root/f'ablation_{tag}_{phase}_metrics.json').read_text('utf-8'))
            values,_=scores(labels[mask],p,len(info['classes']))
            for key,value in values.items():assert abs(value-expected[key])<1e-9
        report['reloaded_models'].append(f'ablation_{tag}')
    report['metrics_and_confusion_matrices_match']=True
    report['ensemble_members']=members
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),'utf-8')
    print(json.dumps({'schema':info['train_schema'],'validation_records':info['n_val'],
       'test_records':info['n_test'],'reloaded_models':report['reloaded_models'],'verified':True}))

if __name__=='__main__':main()
