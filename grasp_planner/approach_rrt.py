"""OPT7: RRT-Connect for the fixed-orientation open-gripper mount translation.

The planner is deliberately generic: the caller supplies the edge validator.
In the grasp pipeline that validator is ``Scene.translation`` (continuous
swept-convex target + environment checks), so no collision model, tolerance or
occupancy is changed here. Every accepted edge was validated by the caller's
function; the shortcut stage also only accepts caller-validated edges.
Robot arm, IK and orientation changes are NOT part of this planner.
"""
import numpy as np

def _steer(a,b,step):
    d=b-a;n=float(np.linalg.norm(d))
    if n<=step:return b.copy(),True
    return a+d*(step/n),False

class _Tree:
    def __init__(self,root):self.nodes=[np.asarray(root,dtype=float)];self.parent=[-1]
    def nearest(self,p):
        return int(np.argmin(np.linalg.norm(np.asarray(self.nodes)-p,axis=1)))
    def add(self,p,parent):self.nodes.append(np.asarray(p,dtype=float));self.parent.append(parent);return len(self.nodes)-1
    def path_to_root(self,i):
        out=[]
        while i!=-1:out.append(self.nodes[i]);i=self.parent[i]
        return out

def rrt_connect(start,goal,edge_free,sample,step_m=.02,max_iterations=300,
                max_edge_checks=400,shortcut_passes=20,rng=None):
    """Return dict(path=list[np.ndarray]|None, iterations, edge_checks, reason).

    ``edge_free(a,b)->bool`` validates the straight segment a->b and is the only
    source of collision truth. ``sample(rng)->point`` draws a configuration.
    The path (when found) goes start -> goal and is deterministic for a seed.
    """
    rng=rng if rng is not None else np.random.default_rng(0)
    start=np.asarray(start,dtype=float);goal=np.asarray(goal,dtype=float)
    checks=[0]
    def free(a,b):
        checks[0]+=1;return bool(edge_free(a,b))
    def result(path,it,reason):
        return dict(path=path,iterations=it,edge_checks=checks[0],reason=reason)
    if free(start,goal):return result([start,goal],0,'direct_edge')
    if not free(start,start):return result(None,0,'start_in_collision')
    if not free(goal,goal):return result(None,0,'goal_in_collision')
    ta,tb=_Tree(start),_Tree(goal);swapped=False
    for it in range(1,max_iterations+1):
        if checks[0]>=max_edge_checks:return result(None,it-1,'edge_check_budget')
        q=np.asarray(sample(rng),dtype=float)
        i=ta.nearest(q);new,_=_steer(ta.nodes[i],q,step_m)
        if not free(ta.nodes[i],new):
            ta,tb=tb,ta;swapped=not swapped;continue
        ia=ta.add(new,i)
        # CONNECT: greedily extend the other tree toward the new node.
        j=tb.nearest(new);reached=False
        while checks[0]<max_edge_checks:
            nxt,done=_steer(tb.nodes[j],new,step_m)
            if not free(tb.nodes[j],nxt):break
            j=tb.add(nxt,j)
            if done:reached=True;break
        if reached:
            pa=ta.path_to_root(ia)[::-1];pb=tb.path_to_root(j)
            path=pa+pb[1:] if np.allclose(pa[-1],pb[0]) else pa+pb
            if swapped:path=path[::-1]
            path=_shortcut(path,free,shortcut_passes,rng)
            return result([np.asarray(x) for x in path],it,'rrt_connect')
        ta,tb=tb,ta;swapped=not swapped
    return result(None,max_iterations,'iteration_budget')

def _shortcut(path,free,passes,rng):
    path=[np.asarray(p,dtype=float) for p in path]
    # Remove duplicates introduced by the tree junction.
    path=[p for k,p in enumerate(path) if k==0 or np.linalg.norm(p-path[k-1])>1e-12]
    for _ in range(passes):
        if len(path)<3:break
        i=int(rng.integers(0,len(path)-2));j=int(rng.integers(i+2,len(path)))
        if free(path[i],path[j]):path=path[:i+1]+path[j:]
    return path
