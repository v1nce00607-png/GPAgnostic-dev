"""Original 3D samples colored by suppression and target label; independent of camera up/down."""
import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

def sample(ids,k,seed=7404):
    ids=np.asarray(ids)
    if len(ids)<=k:return ids
    return np.sort(np.random.default_rng(seed).choice(ids,k,replace=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();src=a.results/'filter_evidence.npz'
    with np.load(src) as z:
        xyz=z['raw_points'];lab=z['new_labels'];keep=z['keep'];deleted=z['halo_deleted'];roi=z['halo_roi'];floatdel=z['floating_deleted']
    cases=[('Háttér – megtartott',keep&(lab==0),'#9399a1',1.2,13000),
           ('Objektumpontok – megtartott',keep&(lab>0),'#119f9c',2.4,8500),
           ('ROI – törölt (feltételezés)',deleted,'#ee433b',3.8,13000),
           ('ROI – védett / megtartott',roi&~deleted,'#e5a62d',2.9,7000),
           ('Depth-ray bizonyítékkal törölt',floatdel,'#8145c8',3.8,4000)]
    traces=[]
    for title,flag,color,size,k in cases:
        inds=sample(np.flatnonzero(flag),k)
        if not len(inds):continue
        pts=xyz[inds]
        traces.append(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',name=f'{title} ({int(np.sum(flag))})',marker=dict(color=color,size=size,opacity=.74)))
    fig=go.Figure(traces)
    fig.update_layout(title='OPT4 – törölt / védett RGB-D pontok, világkoordináták (interaktív 3D)',height=900,scene=dict(aspectmode='data',xaxis_title='X [m]',yaxis_title='Y [m]',zaxis_title='Z [m]'))
    out=a.results/'OPT4_halo_review.html'
    fig.write_html(out,include_plotlyjs=True)
    print('Saved:',out,flush=True)

if __name__=='__main__':main()
