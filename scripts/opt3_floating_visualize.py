"""Create an HTML 3D plot of retained, removed and uncertainty-evidence voxels."""
from pathlib import Path
import argparse
import numpy as np
import plotly.graph_objects as go

def sample(ix,limit,seed=7301):
    ix=np.asarray(ix)
    if len(ix)<=limit:return ix
    return np.sort(np.random.default_rng(seed).choice(ix,limit,replace=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=None);a=ap.parse_args()
    src=a.results/'filter_evidence.npz';z=np.load(src)
    required=('raw_points','new_labels','keep','floating_probe','floating_candidate','floating_deleted')
    missing=[k for k in required if k not in z]
    if missing:raise RuntimeError(f'OPT3 audit arrays not found in {src}: {missing}')
    p=z['raw_points'];l=z['new_labels'];keep=z['keep'];removed=z['floating_deleted']
    traces=[]
    for name,ix,color,size,limit in [
        ('Megtartott háttér',np.flatnonzero(keep&(l==0)),'#91999f',1.5,12000),
        ('Megtartott célobjektum',np.flatnonzero(keep&(l>0)),'#218b96',2.4,8000),
        ('Mélységileg ellentmondásos, eltávolított',np.flatnonzero(removed),'#ec3333',4.2,8000),
        ('Megmaradt izolált jelölt',np.flatnonzero(z['floating_candidate']&keep),'#efbf29',3.0,5000)]:
        ids=sample(ix,limit);xyz=p[ids]
        traces.append(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
            marker=dict(size=size,color=color,opacity=.75),name=f'{name} ({len(ix)})'))
    fig=go.Figure(traces)
    fig.update_layout(title='OPT3: 3D RGB-D foglaltsági bizonyíték — világkoordináták [m]',
        height=850,scene=dict(aspectmode='data',xaxis_title='World X [m]',
        yaxis_title='World Y [m]',zaxis_title='World Z [m]'))
    dest=a.output or a.results/'OPT3_floating_evidence.html'
    fig.write_html(dest,include_plotlyjs=True)
    print(f'Floating evidence visualized: {dest}',flush=True)

if __name__=='__main__':main()
