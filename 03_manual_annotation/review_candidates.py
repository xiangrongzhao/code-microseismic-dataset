"""Prepare human-review labels or copy reviewed candidates into class directories."""
from pathlib import Path
import argparse, json, shutil

CLASSES=['microseismic','blasting','small_landslide','mechanical_mining','transport_vehicles']
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--labels',type=Path,help='Reviewed JSON with one class or reject for each selected filename')
    args=p.parse_args();source=args.candidates.resolve();out=args.out_dir.resolve()
    if out==source or out.is_relative_to(source):p.error('Choose a separate output directory')
    out.mkdir(parents=True,exist_ok=True)
    if args.labels is None:
        dest=out/'annotation_template.json'
        if dest.exists():raise FileExistsError('Existing review labels will not be overwritten: '+str(dest))
        template={'allowed_labels':CLASSES+['reject'],'records':[{'file':f.name,'label':None,'reviewer_note':''} for f in sorted(source.glob('*.npy'))]}
        dest.write_text(json.dumps(template,indent=2),encoding='utf-8')
        print('Manual-review template:',dest)
        return
    labels=json.loads(args.labels.read_text('utf-8'));copied=0
    for row in labels['records']:
        label=row['label']
        if label in [None,'reject']:continue
        if label not in CLASSES:raise ValueError('Unknown label: '+str(label))
        src=(source/row['file']).resolve()
        if not src.is_relative_to(source) or not src.is_file():raise ValueError('Invalid candidate path')
        dest=out/label/src.name;dest.parent.mkdir(exist_ok=True)
        if dest.exists():raise FileExistsError(dest)
        shutil.copy2(src,dest);copied+=1
    print('Reviewed records copied:',copied)
if __name__=='__main__':main()
