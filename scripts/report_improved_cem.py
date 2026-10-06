"""Summarize physical outcomes and provide videos for a completed CEM campaign."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign',type=Path)
    args=p.parse_args()
    root=args.campaign.resolve()
    manifest=json.loads((root/'launch.json').read_text())
    summaries={};all_rows=[]
    for worker in manifest['workers']:
        variant=worker['variant']
        status=json.loads((root/variant/'status.json').read_text())
        rows=status.get('completed_episodes',[])
        summaries[variant]={'status':status['status'],'completed_trials':len(rows),
                            'physical_successes':sum(x['physical_success'] for x in rows),
                            'physical_reached':sum(x['physical_reached'] for x in rows),
                            'physical_safe':sum(x['physical_safe'] for x in rows),
                            'elapsed_seconds':status.get('elapsed_seconds'),
                            'mean_final_goal_error_m':float(np.mean([x['final_goal_error_m'] for x in rows])) if rows else None,
                            'closest_goal_error_m':min([x['minimum_goal_error_m'] for x in rows],default=None),
                            'candidate_evaluations':sum(x['candidate_evaluations'] for x in rows),
                            'trials':rows}
        all_rows.extend([{'variant':variant,**row} for row in rows])
    report={'campaign':str(root),'physical_success_definition':manifest['physical_success'],
            'variants':summaries,'total_physical_successes':sum(x['physical_success'] for x in all_rows),
            'total_completed_trials':len(all_rows),
            'limitations':['Offline simulator labels select and calibrate visual references; runtime planning uses images/actions only.',
                           'GPU1 and GPU2 are demonstration-assisted CEM variants.',
                           'Exact latent STL may be positive while physical reaching fails; physical metrics define success.',
                           'Multiple settings vary between variants; this is not a single-factor ablation.']}
    (root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for col,worker in enumerate(manifest['workers']):
        variant=worker['variant'];rows=summaries[variant]['trials']
        for row in rows:
            path=root/variant/f'scene_{row["scene_seed"]}'/'trajectory.npz'
            with np.load(path) as data:
                goal=data['diagnostic_goal_error']*100
                mug=data['diagnostic_object_displacement']*1000
            t=np.arange(len(goal))/20
            axes[0,col].plot(t,goal,label=str(row['scene_seed']))
            axes[1,col].plot(t,mug)
        axes[0,col].axhline(3,color='black',linestyle='--',label='Reach threshold: 3 cm')
        axes[1,col].axhline(2,color='black',linestyle='--')
        axes[0,col].set_title(variant.replace('gpu','GPU ').replace('_',' '),fontsize=10)
        axes[1,col].set_xlabel('Simulator time (s)')
        axes[0,col].legend(fontsize=7)
    axes[0,0].set_ylabel('Distance to target (cm)')
    axes[1,0].set_ylabel('Mug displacement (mm)')
    fig.tight_layout();fig.savefig(root/'physical_results.png',dpi=150);plt.close(fig)
    cards=[]
    for row in all_rows:
        video=f'{row["variant"]}/scene_{row["scene_seed"]}/trajectory.mp4'
        cards.append(f'<article><h3>{html.escape(row["variant"])} / {row["scene_seed"]}</h3>'
                     f'<p>Physical success: {row["physical_success"]}; mug preserved: {row["physical_safe"]}; '
                     f'closest target distance: {row["minimum_goal_error_m"]*100:.1f} cm.</p>'
                     f'<video controls loop preload="metadata" src="{video}"></video>'
                     f'<p><a href="{video.replace("trajectory.mp4","summary.json")}">Trial metrics</a></p></article>')
    page='''<!doctype html><html><head><meta charset="utf-8"><title>Improved LeWM CEM results</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1200px;margin:32px auto;padding:0 20px}a{color:#93c5fd}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}article{background:#1f2937;padding:16px}video{width:100%;image-rendering:pixelated}img{max-width:100%}</style></head><body>
<h1>Improved LeWM CEM: physical outcomes</h1><p>These are actual CEM simulator executions.
Variants 1 and 2 use training demonstrations to initialize search. Successful latent scores do not establish physical success.</p>
<p><a href="comparison.json">Complete comparison JSON</a> · <a href="launch.json">Run configurations</a></p>
<img src="physical_results.png" alt="Target distance and mug movement during each trial"><div class="grid">'''+''.join(cards)+'</div></body></html>'
    (root/'index.html').write_text(page)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='trials'} for k,v in summaries.items()},indent=2))


if __name__=='__main__':main()
