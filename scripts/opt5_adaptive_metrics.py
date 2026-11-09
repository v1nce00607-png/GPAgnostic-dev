"""Audit the OPT5 CAD-guided experiment without confusing conditional geometry with certification."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse,json


def summarize(rows):
    total=Counter();by_group=defaultdict(Counter);rejects=Counter()
    for r in rows:
        c=r['checks'];status=lambda gate:c.get(gate,{}).get('status')=='PASS'
        g=by_group[(r['hardware'],r['object_instance'])]
        joint=status('target_body') and status('environment')
        adaptive=r.get('adaptive_CAD_closure') or {}
        goal=r.get('opt6_contact_goal_optimization') or {}
        fields={
            'candidates':1,
            'target_body':int(status('target_body')),
            'environment':int(status('environment')),
            'joint_static':int(joint),
            'joint_static_contact_goal':int(joint and r.get('pose_phase','contact_goal')=='contact_goal'),
            'joint_static_open_pregrasp':int(joint and r.get('pose_phase')=='open_pregrasp_staging_no_contact'),
            'dual_CAD_contact':int(adaptive.get('status')=='PASS'),
            'validated_approach':int(joint and status('contact_geometry') and status('approach')),
            'validated_closure':int(joint and status('contact_geometry') and status('closure')),
            'full_model_PASS':int(r['status']=='PASS'),
            'certified_solid_PASS':int(r['status']=='PASS' and r.get('certified_solid_target_collision',False)),
            'conditional_target_surface_PASS':int(status('target_body') and c['target_body'].get('conditional_surface_model',False)),
            'adaptive_attempt':int(bool(adaptive)),
            'adaptive_unverified':int(bool(adaptive) and adaptive.get('status')!='PASS'),
            'opt6_goal_optimizer_attempt':int(bool(goal)),
            'opt6_goal_optimizer_PASS':int(goal.get('status')=='PASS'),
            'opt6_goal_optimizer_near_contact_rejected':int(goal.get('reason')=='near_contact_rejected_by_static_collision'),
            'joint_static_contact_goal_with_dual_contact':int(joint and r.get('pose_phase','contact_goal')=='contact_goal' and adaptive.get('status')=='PASS')}
        for k,n in fields.items():total[k]+=n;g[k]+=n
        if c['target_body'].get('status')=='FAIL':
            rejects[('target_body',c['target_body'].get('cause','unknown'),c['target_body'].get('link','?'))]+=1
        if c['environment'].get('status')=='FAIL':
            rejects[('environment',c['environment'].get('cause','unknown'),c['environment'].get('link','?'))]+=1
        if goal and goal.get('status')!='PASS':
            rejects[('opt6_goal',goal.get('reason','unknown'),
                (goal.get('last_static_failure') or {}).get('link','-'))]+=1
        if adaptive and adaptive.get('status')!='PASS':
            rejects[('adaptive',adaptive.get('reason','unknown'),'-')]+=1
        if c['closure'].get('status')=='FAIL':
            collision=c['closure'].get('target') or c['closure'].get('environment') or {}
            rejects[('closure',collision.get('cause','unknown'),collision.get('link','?'))]+=1
    return total,by_group,rejects


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--new',required=True)
    p.add_argument('--before',required=False)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    rows=json.loads(Path(a.new).read_text(encoding='utf-8'))
    now,groups,rejects=summarize(rows)
    before={}
    if a.before:
        before,_,_=summarize(json.loads(Path(a.before).read_text(encoding='utf-8')))
    keys=['candidates','target_body','environment','joint_static',
          'joint_static_contact_goal','joint_static_open_pregrasp','dual_CAD_contact',
          'validated_approach','validated_closure','full_model_PASS',
          'certified_solid_PASS','conditional_target_surface_PASS',
          'adaptive_attempt','adaptive_unverified',
          'opt6_goal_optimizer_attempt','opt6_goal_optimizer_PASS',
          'opt6_goal_optimizer_near_contact_rejected',
          'joint_static_contact_goal_with_dual_contact']
    lines=[
       '# OPT5 adaptive CAD first-contact – independent gate audit','',
       '> **SCIENTIFIC LIMIT:** `original_surface_ablation` only tests intersections with the **original open target CAD triangles**; it cannot certify unseen solid interior. Full model PASS is **conditional**, not a verified physically executable grasp. The inherited OPT4 aggressive halo deletion is a separate assumption.','',
       '> A Full model PASS means all ten configured geometry gates passed *within this experimental model*. Robot-arm IK, dynamics, force-closure and unobserved environment remain UNVERIFIED.','',
       '| Metric | Previous run | Adaptive run | Delta |','|---|---:|---:|---:|']
    for k in keys:
        b=before.get(k,0)
        lines.append(f'| {k} | {b if before else "n/a"} | {now[k]} | {now[k]-b:+d} |' if before else f'| {k} | n/a | {now[k]} | n/a |')
    lines+=['','## Strict distinction between goals and staging','',
       '- `joint_static_contact_goal`: target_body+environment PASS **at the contact-goal pose**; bilateral contact is an additional required gate.',
       '- `joint_static_contact_goal_with_dual_contact`: same final pose passes BOTH static gates AND bilateral OEM CAD contact.',
       '- `joint_static_open_pregrasp`: true target_body+environment PASS **at a relocated OPEN pregrasp pose**; contact geometry is forced FAIL and the point cannot become full grasp PASS.',
       '- Never compare mixed `joint_static` against older contact-goal-only runs as if they were the same metric.','',
       '## Per gripper and object','','| Gripper | Target | Environment | Target-body | Joint static | Contact-goal static | Open-stage static | OEM dual contact | Approach | Closure | Full model PASS |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (hardware,obj),values in sorted(groups.items()):
        lines.append(f'| {hardware} | {obj} | {values["environment"]} | {values["target_body"]} | {values["joint_static"]} | {values["joint_static_contact_goal"]} | {values["joint_static_open_pregrasp"]} | {values["dual_CAD_contact"]} | {values["validated_approach"]} | {values["validated_closure"]} | {values["full_model_PASS"]} |')
    lines+=['','## Top rejection mechanisms','','| Gate | Cause | Link | Rows |','|---|---|---|---:|']
    for (gate,cause,link),n in rejects.most_common(35):
        lines.append(f'| {gate} | {cause} | {link} | {n} |')
    lines+=['','## OPT6 contact-goal search summary','',
       f'- Attempts: {now["opt6_goal_optimizer_attempt"]}; static/contact PASS: {now["opt6_goal_optimizer_PASS"]}; near-contact rejected by collisions: {now["opt6_goal_optimizer_near_contact_rejected"]}.',
       '- Optimized CAD contacts are not robot-arm or force-closure certificates.',
       '','## Interpretation','',
       '- `dual_CAD_contact` requires both original OEM pads to come within configured tolerance of their corresponding original CAD contacts. It does **not** prove force closure.',
       '- `certified_solid_PASS` remains zero for the inherited experimental ROI and open-CAD surface assumptions.',
       '- If no joint static pass occurs, inspect the target-body failure causes and non-pad link names; do not relax collision gates indiscriminately.',
       '- If dual contact succeeds but closure fails, inspect `checks.closure.target`, `checks.closure.environment` and the recorded segmented trajectory.',
       '- Candidate ranking IDs from different runs do not identify the same physical pose.','']
    output=Path(a.output);output.write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:22]));print('Saved:',output)

if __name__=='__main__':main()
