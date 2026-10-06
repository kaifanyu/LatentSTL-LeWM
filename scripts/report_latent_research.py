"""Audit and summarize saved research experiments without selecting actions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path

import numpy as np


def collect(root):
    rows, jobs = [], []
    directories = sorted(set(root.glob('research_visualpolicy_*')) | set(root.glob('research_waypoint_*')))
    final = root / 'latent_stl_research_20261005T075000Z' / 'final'
    if final.exists():
        directories += sorted(p.parent for p in final.rglob('config.json'))
    for directory in directories:
        if not (directory/'config.json').exists():
            continue
        config = json.loads((directory/'config.json').read_text())
        args = config.get('args', {})
        status = json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {}
        jobs.append({'directory':str(directory.resolve()), 'status':status.get('status'),
                     'args':args, 'error':status.get('error')})
        for file in sorted(directory.glob('scene_*/summary.json')):
            row = json.loads(file.read_text())
            trajectory = file.parent/'trajectory.npz'
            if not trajectory.exists():
                continue
            with np.load(trajectory) as data:
                displacement = data['diagnostic_object_displacement']
                rotation = data['diagnostic_object_rotation_deg']
                distance = data['diagnostic_goal_error']
                safe = bool(np.all(displacement <= .002) and np.all(rotation <= 2.))
                reached = bool(np.any(distance <= .03))
                actual_actions = data['actions'] if 'actions' in data else data['action']
                steps = len(actual_actions)
            assert row['physical_safe'] == safe, file
            assert row['physical_reached'] == reached, file
            assert row['physical_success'] == (safe and reached), file
            role=args.get('evaluation_role','development')
            if role=='heldout':role='untouched_final'
            if args.get('mode') == 'prior':
                method='visual_prior'
            elif args.get('surrogate') == 'final_goal':
                method='final_goal'
            elif args.get('surrogate') == 'waypoint':
                method='waypoint_goal' if args.get('objective')=='goal_only' else 'waypoint_safe'
                if method=='waypoint_safe' and 'temporal_references' not in str(args.get('references','')):
                    method += '_original_refs'
            else:
                method=args.get('variant',directory.name)
            row.update(directory=str(directory.resolve()), summary_path=str(file.resolve()),
                       trajectory_path=str(trajectory.resolve()),
                       variant=args.get('variant',row.get('variant',directory.name)),
                       scenario=args.get('scenario',row.get('scenario','baseline')),
                       evaluation_role=role,
                       method=method,terminal_success=bool(safe and distance[-1]<=.03),
                       held_goal_last_10_steps=bool(safe and np.all(distance[-10:]<=.03)),
                       audited_steps=steps, independent_physical_audit=True)
            rows.append(row)
    return rows, jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path('runs'))
    p.add_argument('--output',type=Path,default=Path('runs/latent_stl_research_20261005T075000Z'))
    args=p.parse_args();output=args.output.resolve();output.mkdir(exist_ok=True,parents=True)
    rows,jobs=collect(args.runs)
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['evaluation_role'],Path(row['directory']).name,row['scenario'])].append(row)
    comparisons=[]
    for (role,name,scenario),trials in grouped.items():
        comparisons.append({'role':role,'run':name,'variant':trials[0]['variant'],'scenario':scenario,
                            'trials':len(trials),'safe_reaches':sum(r['physical_success'] for r in trials),
                            'safe':sum(r['physical_safe'] for r in trials),
                            'reached':sum(r['physical_reached'] for r in trials),
                            'safe_finish':sum(r['terminal_success'] for r in trials),
                            'mean_final_goal_error_cm':float(np.mean([r['final_goal_error_m']*100 for r in trials])),
                            'best_goal_error_cm':min(r['minimum_goal_error_m']*100 for r in trials),
                            'safety_changes_actions_cycles':sum(r.get('safety_veto_changes_actions_cycles',0) for r in trials)})
    final_groups=defaultdict(list)
    for row in rows:
        if row['evaluation_role']=='untouched_final':
            final_groups[(row['method'],row['scenario'])].append(row)
    final_comparison=[]
    for (method,scenario),trials in sorted(final_groups.items()):
        final_comparison.append({'method':method,'scenario':scenario,'trials':len(trials),
                                 'safe_reaches':sum(r['physical_success'] for r in trials),
                                 'safe_finish':sum(r['terminal_success'] for r in trials),
                                 'safe_hold_last_10_steps':sum(r['held_goal_last_10_steps'] for r in trials),
                                 'mug_preserved':sum(r['physical_safe'] for r in trials),
                                 'latent_positive':sum(r.get('actual_stl',-1)>=0 for r in trials),
                                 'mean_final_error_cm':float(np.mean([r['final_goal_error_m']*100 for r in trials]))})
    report={'physical_criteria':{'goal_error_m':.03,'max_mug_displacement_m':.002,'max_mug_rotation_deg':2.,
                                'definition':'Reach at least once while preserving mug throughout fixed budget'},
            'comparisons':comparisons,'final_comparison':final_comparison,'trials':rows,'jobs':jobs,
            'audit':'Success recomputed independently from saved per-step physical measurements, not latent scores.',
            'limitations':['Development trials guide hyperparameter choices and are not an unbiased final test.',
                           'Demonstration-assisted variants use train-only successful trajectories and offline physical labels.',
                           'Safe goal selection is a feasibility-first surrogate; exact latent STL is logged separately.',
                           'Physical success does not establish better calibrated latent predicates or a causal STL benefit.']}
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    if final_comparison:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        methods=['visual_prior','waypoint_goal','waypoint_safe_original_refs','waypoint_safe','final_goal']
        labels=['Visual guide without CEM','Waypoint CEM, no safety','Waypoint CEM, original safety','Waypoint CEM, temporal safety','CEM, final goal only']
        scenarios=['baseline','start_jitter_3cm','mug_x_plus3cm','mug_x_minus3cm','camera_x_plus2cm','dim_085']
        scenario_labels=['Original setup','Start jitter ±3 cm','Mug x +3 cm','Mug x −3 cm','Camera x +2 cm','Brightness ×0.85']
        fig,axes=plt.subplots(1,2,figsize=(16,5.3),sharey=True)
        for ax,key,title in zip(axes,['safe_reaches','safe_finish'],['Safe reach at least once','Safe and at target at final frame']):
            values=np.full((len(methods),len(scenarios)),np.nan)
            counts={}
            for row in final_comparison:
                i,j=methods.index(row['method']),scenarios.index(row['scenario'])
                values[i,j]=row[key]/row['trials'];counts[i,j]=f"{row[key]}/{row['trials']}"
            im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
            for i in range(len(methods)):
                for j in range(len(scenarios)):
                    ax.text(j,i,counts.get((i,j),'—'),ha='center',va='center',
                            color='white' if values[i,j]>.6 else 'black',fontsize=12)
            ax.set_xticks(range(len(scenarios)),scenario_labels,rotation=35,ha='right')
            ax.set_yticks(range(len(methods)),labels);ax.set_title(title)
        fig.suptitle('Frozen final evaluation: counts of successful simulator trials')
        fig.tight_layout();fig.savefig(output/'final_outcomes.png',dpi=160);plt.close(fig)
    head='<tr><th>Role</th><th>Experiment</th><th>Setup</th><th>Safe reaches</th><th>Safe finish</th><th>Mug preserved</th><th>Mean final error</th></tr>'
    body=[]
    for row in comparisons:
        body.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [row['role'],row['run'],row['scenario'],
                    f"{row['safe_reaches']}/{row['trials']}",f"{row['safe_finish']}/{row['trials']}",f"{row['safe']}/{row['trials']}",f"{row['mean_final_goal_error_cm']:.2f} cm"] )+'</tr>')
    cards=[]
    for row in sorted(rows,key=lambda r:(r['evaluation_role']!='untouched_final',not r['physical_success'],r['directory'],r['scene_seed'])):
        video=os.path.relpath(Path(row['summary_path']).parent/'trajectory.mp4',output)
        summary=os.path.relpath(row['summary_path'],output)
        name=f"{row['method']} · {row['scenario']} · {row['scene_seed']}"
        latent_text=f"{row['actual_stl']:.4f}" if 'actual_stl' in row else 'not logged in this earlier run'
        cards.append(f'<article data-search="{html.escape(name+" "+row["evaluation_role"])}"><h3>{html.escape(name)}</h3><p>{html.escape(row["evaluation_role"])} · '
                     f'Safe reach: <strong>{row["physical_success"]}</strong> · Safe finish: {row["terminal_success"]} · '
                     f'Closest: {row["minimum_goal_error_m"]*100:.2f} cm · '
                     f'Final: {row["final_goal_error_m"]*100:.2f} cm</p>'
                     f'<p>Mug preserved: {row["physical_safe"]}; maximum displacement '
                     f'{row["maximum_mug_displacement_m"]*1000:.2f} mm; rotation '
                     f'{row["maximum_mug_rotation_deg"]:.2f}°. Exact observed latent STL: {latent_text}.</p>'
                     f'<video controls loop preload="none" src="{html.escape(video)}"></video>'
                     f'<p><a href="{html.escape(summary)}">Full trial measurements</a></p></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Latent STL research trials</title>
<style>body{font:16px system-ui;color:#e5e7eb;background:#111827;max-width:1500px;margin:32px auto;padding:0 20px}a{color:#93c5fd}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border:1px solid #475569;text-align:left}th{background:#1f2937}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}article{background:#1f2937;padding:16px}video{width:100%;image-rendering:pixelated}p{line-height:1.5}strong{color:#a7f3d0}</style>
<h1>Latent STL: simulator research results</h1>
<p>Physical success means reaching within 3 cm while keeping mug movement below 2 mm and rotation below 2° throughout the trial. Images and past actions guide control; physical state is used only for initialization, offline labels, and evaluation. Successful scores alone are not counted as successful behavior.</p>
<p><a href="comparison.json">Complete results and audit</a> · <a href="protocol.json">Research protocol</a> · <a href="final/freeze.json">Frozen final experiment</a> · <a href="final/amendment_01.json">Added original-threshold control</a></p>
<p><a href="final/waypoint_safe__baseline/scene_910001/trajectory.mp4">Watch a successful held-out CEM trial (4.4 mm final error)</a></p>
<img style="width:100%" src="successful_replay_frames.png" alt="Actual simulator frames showing the arm crossing above the undisturbed mug to the supplied image goal">
<p>Safe reach uses eventual reaching; safe finish additionally requires being within 3 cm at the final frame. The JSON also reports holding within 3 cm over the last 10 frames. Physical safety is required throughout for all three metrics.</p>
<p><a href="waypoint_mechanism.png">Matched waypoint ablation plot</a> · <a href="temporal_references/references.json">Temporal calibration and historical test results</a> · <a href="../dynamics_research_20261005/near_contact_v1/report.json">Near-contact safety diagnostic</a></p>
<p>Development trials are adaptive. Final evaluation configurations are frozen before reserved seeds are executed. Demonstration-assisted methods must be compared against the demonstration baseline before attributing gains to STL.</p><img style="width:100%" src="final_outcomes.png" alt="Final physical success counts">
<h2>What the experiments establish</h2>
<p>Nearby visual waypoints and search around successful training actions produce real safe reaches. In a matched development example, waypoint guidance reaches 0.82 cm from the target; the same model, action prior, and CEM budget using only the final goal stalls 37.6 cm away. Original and multistep-trained dynamics both succeed in four matched development cases, so extra dynamics training was not required for those successes.</p>
<p>The learned model is much less optimistic inside the successful waypoint controller's action distribution: predicted versus actual short-prefix goal gains are about 0.0089 versus 0.0085, compared with 0.0205 versus 0.0027 in the earlier endpoint CEM campaign. This is evidence consistent with better local targets and restricted search reducing model exploitation, rather than proof that a single change explains every gain.</p>
<p>The safety clause still has limits. Whole-trajectory calibration reduces complete-trace false alarms but accepts early disturbance in the targeted collision probe, even when evaluated on real encoded images. Moving an initially undisturbed mug also changes the classifier's response because its references come from the original layout. Positive latent robustness therefore does not establish physical preservation.</p>
<p>Success with a demonstration prior does not by itself establish a benefit from STL: the frozen test includes a prior-only guide and waypoint CEM without safety. Each variation has only three seeds; their counts describe this experiment rather than a precise population success rate.</p>
<p><a href="../dynamics_research_20261005/cem_execution_mechanism_audit.json">Prediction mechanism audit</a> · <a href="calibration_diagnostic_figure/temporal_safety_tradeoff.pdf">Safety calibration figure (PDF)</a> · <a href="final/execution_audit_summary.json">Execution verification</a> · <a href="final/reexecution_verification.json">Saved-action simulator replay check</a></p>
<img style="width:100%" src="waypoint_mechanism.png" alt="Matched waypoint versus final-goal-only development ablation">
<p>Related primary research: <a href="https://arxiv.org/abs/2604.03208">Hierarchical Planning with Latent World Models</a>, <a href="https://arxiv.org/abs/2605.22164">World Model Control by Trajectory Reachability Metrics</a>, and <a href="https://arxiv.org/abs/2608.18746">Decision-Metric Alignment</a>. These motivate the questions; the local results above come from this repository's experiments.</p>
<details><summary>All development and final run measurements</summary><div style="overflow-x:auto"><table>'''+head+''.join(body)+'''</table></div></details><h2>Simulator replays</h2>
<p><label>Filter by method, setup, seed, or role: <input id="filter" type="search" style="padding:8px;width:340px;max-width:90%" placeholder="e.g. waypoint_safe baseline 910001"></label></p>
<div class="grid">'''+''.join(cards)+'''</div><script>
document.getElementById('filter').addEventListener('input',event=>{
 const terms=event.target.value.toLowerCase().trim().split(/\s+/);
 document.querySelectorAll('article[data-search]').forEach(card=>{
  card.hidden=!terms.every(term=>card.dataset.search.toLowerCase().includes(term));
 });
});
</script></html>'''
    (output/'index.html').write_text(page)
    print(json.dumps({'trials':len(rows),'successes':sum(r['physical_success'] for r in rows),
                      'comparisons':comparisons},indent=2))


if __name__=='__main__':main()
