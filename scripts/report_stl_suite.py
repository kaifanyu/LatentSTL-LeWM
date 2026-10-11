"""Build an incremental, artifact-backed HTML report for a two-GPU STL suite.

Usage: python scripts/report_stl_suite.py RUN_ROOT
Only RUN_ROOT is written. Incomplete reports are skipped; completed trial plots
are cached using their source artifact metadata. No GPU or model load is needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path

import h5py
import numpy as np


VERSION = 2
METHODS = {
    "global": "Whole-image projected LeWM latent; cosine distance.",
    "spatial": "LeWM patch features; cosine distances weighted by fixed image-region overlap.",
    "pixels": "Raw RGB root-mean-square difference in the mug and goal regions.",
    "red_pixels": "Red-evidence mug-region difference for preservation; raw RGB goal-region difference for reaching.",
}
SCENARIOS = {
    "baseline": "Original setup, with the ordinary randomized starting pose.",
    "start_jitter_3cm": "Increase starting arm-position jitter to ±3 cm on each axis.",
    "start_x_plus3cm": "Shift the starting arm position +3 cm along x.",
    "mug_x_plus3cm": "Shift the mug +3 cm along x before taking its initial reference.",
    "mug_x_minus3cm": "Shift the mug −3 cm along x before taking its initial reference.",
    "mug_y_plus3cm": "Shift the mug +3 cm along y before taking its initial reference.",
    "camera_x_plus2cm": "Shift the camera +2 cm along x; retain the original goal image.",
    "dim_085": "Multiply observed RGB brightness by 0.85; retain the original goal image.",
}
POLICIES = {
    "noop": "Zero arm command; keep the gripper open. Tests preservation without reaching.",
    "low_motion": "Small sinusoidal translation (up to 2.5 cm per axis) and rotation around the start.",
    "safe_arc": "Move toward the goal with a randomized 10–16 cm upward arc.",
    "collision_straight": "Move directly toward the goal, with no upward or lateral arc.",
    "near_collision": "Move toward the goal with a shallower 2.5–7 cm upward arc.",
    "safe_lateral": "Move toward the goal with a randomized left/right lateral arc of 9–15 cm.",
    "random_smooth": "Hold each random action for eight steps; correct workspace excursions using simulator state.",
    "random_reach": "Randomize the target around the goal and the upward/lateral path offsets.",
    "reach_leave": "Use a safe arc for the first two thirds, then return toward the start; distinguishes eventual reach from terminal reach and sustained reach.",
}
OBJECTIVES = ("terminal", "reach_max", "weighted_penalty", "stl", "reach_and_hold")
STYLE = """body{font:16px/1.55 system-ui,sans-serif;color:#1b293a;background:#f4f6f9;margin:0}
main{max-width:1450px;margin:32px auto;padding:0 24px}h1,h2,h3{line-height:1.2}h2{margin-top:36px}
a{color:#1759a8}table{border-collapse:collapse;width:100%;font-size:14px;background:white}
td,th{padding:9px 12px;border-bottom:1px solid #dde3eb;text-align:left;vertical-align:top}th{background:#e9eef5}
.scroll{overflow:auto}.card,.note{background:white;border:1px solid #dce3ed;border-radius:9px;padding:18px;margin:16px 0}
.note{background:#eaf1fc}.muted{color:#5e6c80}code{background:#edf0f5;padding:2px 4px;overflow-wrap:anywhere}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}.cards .card{margin:0}
.number{font-size:30px;font-weight:650}.trial{display:grid;grid-template-columns:minmax(260px,1fr) minmax(460px,3fr);gap:16px}
img{max-width:100%;height:auto}video{width:100%;max-width:520px}details{margin:10px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.completed,.succeeded{color:#166534}.failed,.blocked{color:#b42318}.running{color:#174fa1}
@media(max-width:800px){.trial{display:block}main{padding:0 12px}td,th{padding:6px}}
"""


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def esc(value):
    return html.escape(str(value), quote=True)


def rel(path, base):
    return esc(os.path.relpath(path, base))


def link(path, base, label=None):
    return f'<a href="{rel(path, base)}">{esc(label or Path(path).name)}</a>'


def pct(value):
    return "—" if value is None else f"{value:.1%}"


def number(value):
    return "—" if value is None else f"{value:.3g}"


def table(headers, rows):
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row)
                                             + '</tr>' for row in rows) + '</tbody></table></div>')


def page(title, body, refresh=False):
    meta = '<meta http-equiv="refresh" content="60">' if refresh else ''
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{meta}<title>{esc(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>'


def smoke_notice(protocol):
    if not protocol.get('smoke'):
        return ''
    steps = protocol.get('training_steps_per_seed', 2)
    fine = protocol.get('finetune_steps_per_seed', 2)
    return f'<div class="note"><strong>Infrastructure smoke test — {steps} training steps and {fine} fine-tuning steps per seed. These are not research estimates.</strong> This small run checks GPU execution, simulation, calibration, evaluation and report generation. Use the full suite for substantive comparisons.</div>'


def fingerprint(paths):
    facts = [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.exists()]
    return hashlib.sha256(json.dumps([VERSION, facts]).encode()).hexdigest()


def aggregate(records):
    """Pool counts, not unweighted scenario percentages; retain seed and duration."""
    groups = defaultdict(list)
    for row in records:
        groups[(row['model_seed'], row['steps'], row['method'], row['operating_point'])].append(row)
    result = []
    for (seed, steps, method, op), rows in sorted(groups.items(), key=lambda x: str(x[0])):
        count = sum(r['episodes'] for r in rows)
        safe_positive = sum(r['frame_safe']['positives'] for r in rows)
        safe_negative = sum(r['frame_safe']['negatives'] for r in rows)
        ratio = lambda a, b: a / b if b else None
        ranks = {obj: {key: sum(r['offline_candidate_ranking'][obj][key] for r in rows)
                       for key in ('banks', 'safe_reach', 'unsafe', 'safe_hold')} for obj in OBJECTIVES}
        result.append({'model_seed': seed, 'steps': steps, 'method': method, 'operating_point': op,
                       'shards': len(rows), 'episodes': count,
                       'frame_false_safe': ratio(sum(r['frame_safe']['false_safe_count'] for r in rows), safe_negative),
                       'frame_false_unsafe': ratio(sum(r['frame_safe']['false_unsafe_count'] for r in rows), safe_positive),
                       'stl_agreement': ratio(sum(r['trajectory_stl']['sign_agreement'] * r['episodes'] for r in rows), count),
                       'hold_agreement': ratio(sum(r['trajectory_hold']['sign_agreement'] * r['episodes'] for r in rows), count),
                       'missed_violations': sum(r['missed_violations'] for r in rows),
                       'unsafe_episodes': sum(r['unsafe_episodes'] for r in rows), 'ranking': ranks})
    return result


def describe_protocol(protocol):
    waves = protocol.get('waves', [])
    tolerances = protocol.get('physical_tolerances', {})
    displacement = tolerances.get('displacement_m', .002)
    rotation = tolerances.get('rotation_deg', 2.)
    goal = tolerances.get('goal_m', .03)
    hz = protocol.get('frame_freq_hz', 20)
    body = '<h2>How a trial is conducted</h2><ol>'
    body += '<li>Collect disjoint training, validation and test candidate banks. A bank contains multiple scripted policies replayed from exactly the same simulator snapshot. The collection controller can use simulator state.</li>'
    body += '<li>Train the LeWM representation with each declared random seed. Fit two scalar distance thresholds on validation labels only; freeze them before fresh scenarios are evaluated.</li>'
    body += '<li>For each scenario and bank, record all eight base policies plus <code>reach_leave</code>, which reaches and then returns toward the start. All representations inspect the same recorded RGB trajectories. Independent scene seeds generate later waves.</li>'
    body += '<li>Compute image distances frame by frame, convert them to signed margins, and evaluate preservation, eventual reach and reach-and-hold. Compare their signs with physical labels computed from simulator poses. Rank the nine already-recorded candidates within each bank.</li></ol>'
    if waves:
        body += table(['Wave', 'First scene seed', 'Banks / scenario', 'Control steps', 'Simulated duration'],
                      [[esc(w.get('name')), esc(w.get('seed')), esc(w.get('banks')), esc(w.get('steps')),
                        f"{w.get('steps', 0) / hz:g} s"] for w in waves])
    body += f'<p>Images are sampled at {hz:g} Hz; a trajectory includes its initial frame. Simulator safety requires displacement ≤ {displacement * 1000:g} mm and rotation ≤ {rotation:g}°. Reach requires end-effector distance ≤ {goal * 100:g} cm. These physical measurements are evaluation labels and validation calibration supervision, not runtime image-predicate inputs.</p>'
    body += f'<p>A 10-observation hold spans nine sampling intervals ({9 / hz:g} s between its first and last observations). The 192-step wave runs the scripts for longer and changes their action timing; it is a separate duration stress test, not a simple continuation of the same 96-step trajectory.</p>'
    body += '<h3>Representations and references</h3>' + table(['Method', 'Distance representation'], [[esc(k), esc(v)] for k, v in METHODS.items()])
    body += '<p>The safety reference is each episode’s initial image. The original supplied goal image remains fixed under every scenario. Mug regions are detected once from the initial RGB using an explicit red-color prior; they do not follow later motion. The goal region is fixed. Region-detection fallback counts are reported.</p>'
    body += '<h3>Predicate formulas and comparisons</h3><p>With validation-frozen threshold θ and scale σ, <code>s(t) = (θ_safe − d_safe(t)) / σ_safe</code> and <code>g(t) = (θ_goal − d_goal(t)) / σ_goal</code>. Nonnegative margins satisfy the predicate. These are normalized margins, not probabilities.</p>'
    body += table(['Objective', 'Sampled-time score', 'Meaning'], [
        ['Always preserve', '<code>min_t s(t)</code>', 'Every observed frame must preserve the mug.'],
        ['Eventually reach', '<code>max_t g(t)</code>', 'At least one observed frame must reach.'],
        ['STL preserve ∧ reach', '<code>min(min_t s(t), max_t g(t))</code>', 'Preserve throughout and reach at least once.'],
        ['STL reach and hold', '<code>min(min_t s(t), max_u min_{t=u…u+9} g(t))</code>', 'Preserve throughout; reach for 10 consecutive sampled frames.'],
        ['Terminal baseline', '<code>g(T)</code>', 'Compare terminal goal similarity; preservation is not constrained.'],
        ['Weighted-penalty baseline', '<code>max_t g(t) − mean_t max(−s(t), 0)</code>', 'Soft tradeoff baseline; does not enforce the STL conjunction.'],
    ])
    body += '<p><b>Balanced</b> thresholds optimize validation balanced accuracy. <b>false_safe_05</b> selects the largest preservation threshold accepting at most 5% of physically unsafe validation frames. Both share the goal threshold and scales. The 5% validation constraint is not a guarantee on fresh test scenes.</p>'
    body += '<h3>Scenario changes</h3>' + table(['Scenario', 'Change'], [[esc(s), esc(SCENARIOS.get(s, 'See frozen protocol.'))] for s in protocol.get('scenarios', SCENARIOS)])
    body += '<h3>Scripted policies in each fresh bank</h3><p>Policy names describe intended behavior. Physical success is measured after simulation; a policy called “safe” is not assumed to have preserved the mug.</p>' + table(['Policy', 'How actions are generated'], [[esc(k), esc(v)] for k, v in POLICIES.items()])
    body += '<div class="note"><b>Scope and limits.</b> This is observed RGB monitoring and offline candidate ranking. Scripted actions are not generated by an STL planner. Forecast audits test the model on recorded actions and are reported separately. Spatial/pixel predicates are not claimed to operate on predicted global latents. Static pose-intervention probes are excluded from trajectory counts and temporal-logic evaluations. Candidate frames share initial-state banks; individual reports give confidence intervals resampled by bank. Partial aggregates compare only completed shards, so wait for matching coverage before judging methods. RGB baselines can be identical across model seeds because they do not use learned features.</div>'
    return body


def plot_episode(rec, operating_points, path, title, hz, tolerances):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    time = np.arange(len(rec['safe_distance'])) / hz
    for ax, (op, cfg) in zip(axes[:2], operating_points.items()):
        safe = (cfg['safe_threshold'] - rec['safe_distance']) / cfg['safe_scale']
        reach = (cfg['goal_threshold'] - rec['goal_distance']) / cfg['goal_scale']
        ax.plot(time, safe, color='#167047', label='preserve margin s(t)')
        ax.plot(time, reach, color='#245eaf', label='reach margin g(t)')
        ax.axhline(0, color='#3f4650', lw=.8, ls='--')
        ax.fill_between(time, 0, 1, where=~rec['safe_label'].astype(bool), transform=ax.get_xaxis_transform(),
                        color='#dd4f43', alpha=.12, label='physically unsafe frame')
        ax.set_ylabel('signed margin'); ax.set_title(op, loc='left', fontsize=10)
        ax.legend(loc='best', fontsize=8, ncol=3); ax.grid(alpha=.2)
    axes[2].plot(time, rec['object_displacement'] / tolerances.get('displacement_m', .002), label='displacement / tolerance')
    axes[2].plot(time, rec['object_rotation_deg'] / tolerances.get('rotation_deg', 2.), label='rotation / tolerance')
    axes[2].plot(time, rec['goal_error'] / tolerances.get('goal_m', .03), label='goal error / tolerance')
    axes[2].axhline(1, color='#3f4650', lw=.8, ls='--')
    axes[2].set_ylabel('physical diagnostic / tolerance'); axes[2].set_xlabel('simulated time (s)')
    axes[2].legend(fontsize=8, ncol=3); axes[2].grid(alpha=.2)
    fig.suptitle(title, fontsize=11); fig.tight_layout()
    tmp = path.with_name(path.name + '.tmp')
    fig.savefig(tmp, format='png', dpi=120); plt.close(fig); tmp.replace(path)


def trial_report(root, report_path, report, protocol, seed, shard, steps):
    out = report_path.parent
    trace = out / 'test_traces.h5'
    dependencies = [report_path, trace, root / 'protocol.json']
    dependencies += [out / f'episodes_{op}.json' for op in report['operating_points']]
    stamp = fingerprint(dependencies)
    cached = read_json(out / 'visualization_cache.json', {})
    if cached.get('fingerprint') == stamp and (out / 'index.html').exists():
        return True
    if not trace.exists():
        return False
    with h5py.File(trace, 'r') as h:
        if not h.attrs.get('complete', False):
            return False
        keys = sorted(h.keys())
        if not keys:
            return False
        first_bank = h[keys[0]].attrs['bank']
        examples = []
        for key in keys:
            ep = h[key]
            if ep.attrs['bank'] != first_bank:
                continue
            examples.append((key, {name: ep[name][:] for name in ep.keys()}, dict(ep.attrs)))
    data_dir = root / 'data' / shard
    title = f'Seed {seed} · {shard} · {report["method"]}'
    body = f'<p>{link(root / "index.html", out, "← Suite overview")}</p><h1>{esc(title)}</h1>' + smoke_notice(protocol)
    body += f'<p>{esc(METHODS.get(report["method"], report["method"]))} {steps} control steps. {len(examples)} first-bank examples below; metrics use every completed bank.</p>'
    body += '<p>' + ' · '.join(link(p, out) for p in (report_path, trace, data_dir / 'audit.json', data_dir / 'collection.json') if p.exists()) + '</p>'
    rows = []
    for op, r in report['operating_points'].items():
        ci = r.get('bank_bootstrap_95ci', {})
        interval = ci.get('stl_agreement')
        rows.append([esc(op), str(r['episodes']), pct(r['frame_safe']['false_safe_rate']), pct(r['frame_safe']['false_unsafe_rate']),
                     pct(r['trajectory_stl']['sign_agreement']), pct(r['trajectory_hold']['sign_agreement']),
                     f"{r['missed_violations']}/{r['unsafe_episodes']}",
                     '—' if interval is None else f'{pct(interval[0])}–{pct(interval[1])}'])
    body += table(['Operating point', 'Episodes', 'False safe ↓', 'False unsafe ↓', 'STL agreement ↑', 'Hold agreement ↑', 'Missed violations ↓', 'STL bank-bootstrap 95% interval'], rows)
    body += '<p>False safe counts physically unsafe frames accepted as safe. False unsafe counts physically safe frames rejected. Detection delay begins at the first physical violation and is conditional on eventual detection; misses are shown separately. A temporal score ≥ 0 predicts satisfaction.</p>'
    body += '<h2>Frozen thresholds</h2>' + table(['Operating point', 'Safe θ', 'Goal θ', 'Safe σ', 'Goal σ'], [[esc(op)] + [number(cfg[k]) for k in ('safe_threshold', 'goal_threshold', 'safe_scale', 'goal_scale')] for op, cfg in report['calibration']['operating_points'].items()])
    body += '<h2>Offline ranking of the same candidate bank</h2>'
    rankings = []
    for op, r in report['operating_points'].items():
        for objective, rank in r['offline_candidate_ranking'].items():
            rankings.append([esc(op), esc(objective), str(rank['banks']), str(rank['safe_reach']), str(rank['unsafe']), str(rank['safe_hold'])])
    body += table(['Operating point', 'Ranking objective', 'Banks', 'Selected safe reach ↑', 'Selected unsafe ↓', 'Selected safe hold ↑'], rankings)
    body += '<h2>Watch one bank, then inspect its predicate traces</h2><p>Each video is the actual simulator rollout. Plot rows show the two operating points, followed by physical diagnostic values divided by their tolerances. Shaded red regions mark physically unsafe frames. The dashed physical threshold is 1; the predicate boundary is 0. Later banks are in the linked HDF5 file.</p>'
    episode_rows = {op: {x['id']: x for x in read_json(out / f'episodes_{op}.json', [])} for op in report['operating_points']}
    for key, rec, attrs in examples:
        plot = out / f'trace_{key}.png'
        plot_episode(rec, report['calibration']['operating_points'], plot, f"{attrs['policy']} · bank {first_bank}", protocol.get('frame_freq_hz', 20), protocol.get('physical_tolerances', {}))
        video = data_dir / f"{attrs['policy']}.mp4"
        video_html = f'<video controls preload="none" src="{rel(video, out)}"></video>' if video.exists() else '<p>Video unavailable.</p>'
        body += f'<section class="card"><h3>{esc(attrs["policy"])}</h3><div class="trial"><div>{video_html}<p>Episode <code>{esc(key)}</code><br>Physical safe: <b>{bool(rec["safe_label"].all())}</b><br>Physical reach: <b>{bool(rec["reach_label"].any())}</b></p></div><a href="{rel(plot, out)}"><img loading="lazy" src="{rel(plot, out)}" alt="Preserve and reach margins with physical diagnostics"></a></div>'
        erows = []
        for op, by_id in episode_rows.items():
            r = by_id.get(key, {})
            erows.append([esc(op)] + [number(r.get(k)) for k in ('safe_min', 'reach_max', 'stl', 'reach_and_hold')]
                         + [esc(r.get('detection_delay_frames')), esc(r.get('missed_violation')), esc(r.get('pre_violation_false_alarm'))])
        body += table(['Operating point', 'G preserve', 'F reach', 'STL conjunction', 'Reach & hold', 'Alarm delay (frames)', 'Missed violation', 'Prior false alarm'], erows) + '</section>'
    probe = report.get('static_observability', {})
    if probe:
        body += '<h2>Static observability probes</h2><p>Artificial pose changes test visibility at and around the physical tolerances. They are not simulated action trajectories and are excluded from temporal metrics.</p>'
        body += table(['Axis', 'Applied amount (m or °)', 'Physical safe', 'Visual safe distance', 'Balanced predicts safe', 'Constrained predicts safe'],
                      [[esc(r['axis']), number(r['amount']), esc(r['safe_label']), number(r['safe_distance']), esc(r['predicted_safe'].get('balanced')), esc(r['predicted_safe'].get('false_safe_05'))] for r in probe.get('rows', [])])
    body += '<p>Full formulas, reference inputs, scenario definitions, and limitations appear in the suite overview. Checkpoint SHA-256: <code>' + esc(report['calibration'].get('checkpoint_sha256')) + '</code>.</p>'
    atomic(out / 'index.html', page(title, body))
    atomic(out / 'visualization_cache.json', json.dumps({'fingerprint': stamp, 'version': VERSION}))
    return True


def comparison_plot(root, aggregates, smoke=False):
    if not aggregates:
        return
    digest = hashlib.sha256(json.dumps([aggregates, smoke], sort_keys=True).encode()).hexdigest()
    cache = root / 'comparison_cache.json'
    if read_json(cache, {}).get('digest') == digest and (root / 'comparison.png').exists():
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = sorted({(r['model_seed'], r['steps']) for r in aggregates}, key=str)
    fig, axes = plt.subplots(len(groups), 3, figsize=(15, 3.6 * len(groups)), squeeze=False)
    for axs, (seed, steps) in zip(axes, groups):
        rows = [r for r in aggregates if (r['model_seed'], r['steps']) == (seed, steps)]
        labels = list(METHODS)
        for ax, metric, title in zip(axs, ('frame_false_safe', 'stl_agreement', 'hold_agreement'), ('Unsafe frames accepted ↓', 'STL agreement ↑', 'Hold agreement ↑')):
            for offset, op in ((-.19, 'balanced'), (.19, 'false_safe_05')):
                values = [next((r[metric] for r in rows if r['method'] == m and r['operating_point'] == op), None) for m in labels]
                ax.bar(np.arange(len(labels)) + offset, [np.nan if v is None else v for v in values], .36, label=op)
            ax.set_xticks(np.arange(len(labels)), labels, rotation=20); ax.set_ylim(0, 1)
            ax.set_title(f'Seed {seed} · {steps} steps | {title}', fontsize=10); ax.grid(axis='y', alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle('Infrastructure smoke test — not research estimates' if smoke else 'Completed fresh trials; seeds and trajectory durations kept separate', fontsize=12)
    fig.tight_layout(); tmp = root / 'comparison.png.tmp'; fig.savefig(tmp, format='png', dpi=150); plt.close(fig)
    tmp.replace(root / 'comparison.png'); atomic(cache, json.dumps({'digest': digest}))


def main(root):
    root = Path(root).resolve()
    protocol = read_json(root / 'protocol.json', {})
    status = read_json(root / 'status.json', {})
    waves = {w['name']: w for w in protocol.get('waves', [])}
    collected, records, trials, warnings = [], [], [], []
    for path in sorted((root / 'data').glob('*/collection.json')):
        item = read_json(path)
        if item and item.get('complete'):
            collected.append({'shard': path.parent.name, 'episodes': len(item['episodes']), 'seconds': item.get('seconds')})
    for path in sorted((root / 'models').glob('seed_*/evaluations/*/*/report.json')):
        report = read_json(path)
        if not report or not report.get('operating_points'):
            continue
        seed = path.parents[3].name.removeprefix('seed_')
        shard = path.parents[1].name
        wave = shard.split('__', 1)[0]
        collection = read_json(root / 'data' / shard / 'collection.json', {})
        steps = collection.get('args', {}).get('steps', waves.get(wave, {}).get('steps'))
        for op, result in report['operating_points'].items():
            records.append({'model_seed': seed, 'steps': steps, 'shard': shard, 'method': report['method'],
                            'operating_point': op, 'report': str(path.relative_to(root)), **result})
        try:
            ready = trial_report(root, path, report, protocol, seed, shard, steps)
        except (OSError, KeyError, ValueError) as error:
            warnings.append(f'{path.relative_to(root)}: {type(error).__name__}: {error}')
            ready = False
        trials.append({'model_seed': seed, 'shard': shard, 'steps': steps, 'method': report['method'],
                       'path': str((path.parent / ('index.html' if ready else 'report.json')).relative_to(root)),
                       'episodes': next(iter(report['operating_points'].values()))['episodes']})
    aggregates = aggregate(records)
    comparison_plot(root, aggregates, smoke=protocol.get('smoke', False))
    updated = datetime.now(timezone.utc).isoformat(timespec='seconds')
    summary = {'updated_at_utc': updated, 'fresh_trajectories_completed': sum(r['episodes'] for r in collected),
               'completed_shards': collected, 'aggregates': aggregates, 'by_shard': records, 'report_warnings': warnings,
               'scope': 'Observed RGB monitoring and offline candidate ranking; scripted collection is not action planning.'}
    atomic(root / 'summary.json', json.dumps(summary, indent=2) + '\n')
    jobs = status.get('jobs', [])
    if isinstance(jobs, dict):
        jobs = [{'name': name, **row} for name, row in jobs.items()]
    counts = Counter(j.get('status', 'unknown') for j in jobs)
    body = '<h1>LatentSTL · two-GPU experiment suite</h1>' + smoke_notice(protocol)
    body += f'<p class="muted">Updated {esc(updated)} · page refreshes every 60 seconds · overall status: <b>{esc(status.get("status", "awaiting supervisor"))}</b></p>'
    body += '<div class="cards">' + ''.join(f'<div class="card"><div class="number">{value}</div>{label}</div>' for value, label in [
        (f'{summary["fresh_trajectories_completed"]} / {protocol.get("fresh_trajectories", "?")}', 'fresh simulated trajectories collected'),
        (len(trials), 'completed representation/scenario evaluations'),
        (counts.get('running', 0), 'jobs running'),
        (counts.get('failed', 0), 'failed jobs')]) + '</div>'
    body += '<p>' + ' · '.join(link(root / name, root, label) for name, label in [('TRIALS.md', 'Detailed trial guide'), ('protocol.json', 'Frozen protocol and exact commands'), ('status.json', 'Live supervisor status'), ('summary.json', 'Machine-readable results'), ('reference_audit.json', 'Reference and scene-seed audit'), ('frozen_calibration.json', 'Frozen calibration hashes')] if (root / name).exists()) + '</p>'
    frozen = sorted((root / 'models').glob('seed_*/frozen_calibration.json'))
    if frozen:
        body += '<p>Frozen calibration hashes: ' + ' · '.join(link(path, root, path.parent.name) for path in frozen) + '</p>'
    body += '<p class="note">Start with a completed trial below: play a policy video, follow its preserve/reach curves, and compare the two calibrated operating points. Results appear only after their files complete. Training and simulation can take hours; this page shows their current progress.</p>'
    body += '<h2>Live jobs and training progress</h2>'
    active_rows, other_rows = [], []
    for j in sorted(jobs, key=lambda j: (j.get('status') not in ('running', 'failed'), j.get('name', ''))):
        log = Path(j.get('log') or '')
        if not log.is_absolute():
            log = root / log
        command = j.get('command', '')
        if isinstance(command, list):
            import shlex
            command = shlex.join(str(x) for x in command)
        details = f'<details><summary>Command</summary><pre>{esc(command)}</pre></details>' if command else ''
        row = [esc(j.get('name')), f'<span class="{esc(j.get("status"))}">{esc(j.get("status"))}</span>',
               esc(j.get('gpu', '—')), esc(j.get('pid', '—')), link(log, root, 'log') if log.is_file() else '—', details]
        (active_rows if j.get('status') in ('running', 'failed', 'blocked') else other_rows).append(row)
    headers = ['Job', 'Status', 'GPU', 'PID', 'Output', 'Invocation']
    body += table(headers, active_rows) if active_rows else '<p>No running or failed jobs currently reported.</p>'
    if other_rows:
        body += f'<details><summary>All other jobs ({len(other_rows)} queued/completed)</summary>' + table(headers, other_rows) + '</details>'
    for path in sorted((root / 'models').glob('seed_*/*/status.json')):
        state = read_json(path, {})
        body += f'<details><summary>{esc(path.parent.parent.name)} / {esc(path.parent.name)} · step {esc(state.get("global_step", state.get("step", "—")))}</summary><pre>{esc(json.dumps(state, indent=2))}</pre></details>'
    collection_progress = []
    for path in sorted((root / 'data').glob('*/progress.jsonl')) + sorted((root / 'dataset').glob('progress.jsonl')):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        collection_progress.append([esc(path.parent.name), str(len(lines)), link(path, root, 'episode log')])
    if collection_progress:
        body += '<details><summary>Collection progress by dataset</summary>' + table(['Dataset', 'Episode records written', 'Details'], collection_progress) + '</details>'
    gallery = []
    for directory in sorted((root / 'data').glob('*')):
        if not directory.is_dir():
            continue
        videos = sorted(directory.glob('*.mp4'))
        if not videos:
            continue
        # Progress is written before video encoding. Until final publication,
        # exclude the latest policy so an unfinalized MP4 is not advertised.
        if not read_json(directory / 'collection.json', {}).get('complete'):
            try:
                events = [json.loads(line) for line in (directory / 'progress.jsonl').read_text().splitlines() if line.strip()]
            except (OSError, ValueError):
                events = []
            finished = {row.get('policy') for row in events[:-1]}
            videos = [video for video in videos if video.stem in finished]
        if videos:
            gallery.append(f'<details><summary>{esc(directory.name)} · {len(videos)} first-bank videos</summary><div class="cards">'
                           + ''.join(f'<div class="card"><b>{esc(video.stem)}</b><video controls preload="none" src="{rel(video, root)}"></video><p>{esc(POLICIES.get(video.stem, ""))}</p></div>' for video in videos)
                           + '</div></details>')
    if gallery:
        body += '<h2>Simulation videos available now</h2><p>First-bank rollouts appear here during collection, before training or predicate evaluation finishes. Physical success and predicate comparisons appear in the completed trial viewers.</p>' + ''.join(gallery)
    body += '<h2>Completed trial viewers</h2>'
    body += table(['Model seed', 'Scenario / wave', 'Steps', 'Representation', 'Episodes', 'Viewer'],
                  [[esc(r['model_seed']), esc(r['shard']), esc(r['steps']), esc(r['method']), str(r['episodes']), link(root / r['path'], root, 'video + predicate traces')] for r in trials]) if trials else '<p>No completed fresh evaluations yet. Training and collection logs above show current progress.</p>'
    body += '<h2>Comparison on completed fresh data</h2><p>Rates pool raw counts across completed shards of the same duration. Model seeds and trajectory lengths are kept separate. Coverage is shown explicitly; missing methods have no results yet. Aggregate bars do not show confidence intervals; bank-bootstrap intervals are available in each trial.</p>'
    body += table(['Seed', 'Steps', 'Method', 'Operating point', 'Shards', 'Episodes', 'False safe ↓', 'False unsafe ↓', 'STL agree ↑', 'Hold agree ↑', 'Missed / unsafe'],
                  [[esc(r['model_seed']), esc(r['steps']), esc(r['method']), esc(r['operating_point']), str(r['shards']), str(r['episodes']), pct(r['frame_false_safe']), pct(r['frame_false_unsafe']), pct(r['stl_agreement']), pct(r['hold_agreement']), f"{r['missed_violations']}/{r['unsafe_episodes']}"] for r in aggregates]) if aggregates else '<p>Awaiting completed evaluations.</p>'
    if aggregates:
        body += '<p><a href="comparison.png"><img src="comparison.png" alt="Per-seed, per-duration comparison of predicate methods"></a></p>'
    body += '<h2>Forecast and calibration diagnostics</h2><p>Recorded-action world-model forecasts are separate from observed visual monitoring. Calibration reports also include the source dataset’s held-out test split; fresh scenarios above remain distinct.</p><ul>'
    for pattern in ('seed_*/calibration/*/report.json', 'seed_*/forecasts/*/report.json'):
        for path in sorted((root / 'models').glob(pattern)):
            label = str(path.parent.relative_to(root / 'models'))
            body += '<li>' + link(path, root, label)
            plots = list(path.parent.glob('*.png'))
            if plots:
                body += ' · ' + ' · '.join(link(p, root) for p in plots)
            body += '</li>'
    body += '</ul>' + describe_protocol(protocol)
    if warnings:
        body += '<h2>Report rendering issues</h2><p>Raw reports remain linked; the next refresh will retry visualization.</p><pre>' + esc('\n'.join(warnings)) + '</pre>'
    atomic(root / 'index.html', page('LatentSTL two-GPU suite', body, refresh=True))
    print(json.dumps({'report': str(root / 'index.html'), 'trials': len(trials), 'fresh_trajectories': summary['fresh_trajectories_completed'], 'warnings': warnings}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
