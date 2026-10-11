"""Check report aggregation, incomplete-run behavior and cached trial rendering."""
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'report_stl_suite.py'
spec = importlib.util.spec_from_file_location('stl_suite_report', SCRIPT)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def metrics(episodes=2, positives=8, negatives=2, false_safe=1):
    cls = {'positives': positives, 'negatives': negatives,
           'false_safe_count': false_safe, 'false_unsafe_count': 0,
           'false_safe_rate': false_safe / negatives, 'false_unsafe_rate': 0.,
           'sign_agreement': .5}
    return {'episodes': episodes, 'frame_safe': cls, 'trajectory_stl': cls,
            'trajectory_hold': cls, 'missed_violations': 0, 'unsafe_episodes': 1,
            'offline_candidate_ranking': {key: {'banks': 1, 'safe_reach': 1, 'unsafe': 0, 'safe_hold': 0}
                                          for key in reporter.OBJECTIVES}}


def test_aggregate_uses_counts_and_keeps_seeds_and_durations_separate():
    base = {'model_seed': '0', 'steps': 96, 'method': 'spatial', 'operating_point': 'balanced'}
    a = {**base, **metrics(negatives=2, false_safe=1)}
    b = {**base, **metrics(negatives=8, false_safe=1)}
    longer = {**a, 'steps': 192}
    other_seed = {**a, 'model_seed': '1'}
    result = reporter.aggregate([a, b, longer, other_seed])
    assert len(result) == 3
    pooled = next(x for x in result if x['model_seed'] == '0' and x['steps'] == 96)
    assert pooled['frame_false_safe'] == .2
    assert pooled['episodes'] == 4
    assert pooled['ranking']['stl']['banks'] == 2


def test_empty_and_incomplete_run_does_not_invent_results(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps({'fresh_trajectories': 1728}))
    partial = tmp_path / 'models' / 'seed_0' / 'evaluations' / 'wave1__baseline' / 'spatial'
    partial.mkdir(parents=True)
    (partial / 'report.json').write_text('{"method":')
    reporter.main(tmp_path)
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert result['fresh_trajectories_completed'] == 0
    assert result['aggregates'] == []
    assert not (tmp_path / 'comparison.png').exists()
    assert 'No completed fresh evaluations yet' in (tmp_path / 'index.html').read_text()


def test_smoke_notice_job_collapsing_and_only_finalized_early_videos(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps({'smoke': True, 'fresh_trajectories': 18}))
    (tmp_path / 'status.json').write_text(json.dumps({'jobs': [
        {'name': 'active', 'status': 'running', 'gpu': 0},
        {'name': 'later', 'status': 'queued'}, {'name': 'earlier', 'status': 'completed'}]}))
    (tmp_path / 'TRIALS.md').write_text('Detailed protocol')
    directory = tmp_path / 'data' / 'smoke__baseline'
    directory.mkdir(parents=True)
    (directory / 'noop.mp4').touch()
    (directory / 'low_motion.mp4').touch()
    (directory / 'progress.jsonl').write_text('\n'.join(json.dumps({'policy': policy}) for policy in ('noop', 'low_motion')))
    reporter.main(tmp_path)
    page = (tmp_path / 'index.html').read_text()
    assert 'Infrastructure smoke test' in page and 'not research estimates' in page
    assert '2 training steps and 2 fine-tuning steps' in page
    assert 'All other jobs (2 queued/completed)' in page
    assert 'Detailed trial guide' in page
    assert 'smoke__baseline/noop.mp4' in page
    assert 'smoke__baseline/low_motion.mp4' not in page
    assert '0.45 s' in page and 'changes their action timing' in page


def test_trial_plot_links_and_incremental_cache(tmp_path):
    out = tmp_path / 'models' / 'seed_0' / 'evaluations' / 'wave1__baseline' / 'spatial'
    out.mkdir(parents=True)
    protocol = {'waves': [{'name': 'wave1', 'steps': 96, 'banks': 1, 'seed': 900}],
                'fresh_trajectories': 9, 'scenarios': ['baseline']}
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    cfg = {'safe_threshold': .1, 'goal_threshold': .2, 'safe_scale': .03, 'goal_scale': .04}
    report = {'method': 'spatial', 'calibration': {'checkpoint_sha256': 'fixture',
              'operating_points': {'balanced': cfg, 'false_safe_05': cfg}},
              'operating_points': {'balanced': metrics(), 'false_safe_05': metrics()}}
    (out / 'report.json').write_text(json.dumps(report))
    video = tmp_path / 'data' / 'wave1__baseline' / 'safe_arc.mp4'
    video.parent.mkdir(parents=True)
    video.touch()
    with h5py.File(out / 'test_traces.h5', 'w') as h:
        h.attrs['complete'] = True
        ep = h.create_group('000_safe_arc')
        ep.attrs['bank'] = 'baseline__900'
        ep.attrs['policy'] = 'safe_arc'
        for name, values in {'safe_distance': [.0, .01, .2], 'goal_distance': [.5, .3, .1],
                             'safe_label': [True, True, False], 'reach_label': [False, False, True],
                             'object_displacement': [0, .001, .003], 'object_rotation_deg': [0, 0, 0],
                             'goal_error': [.1, .05, .01]}.items():
            ep.create_dataset(name, data=np.asarray(values))
    for op in report['operating_points']:
        (out / f'episodes_{op}.json').write_text(json.dumps([{'id': '000_safe_arc', 'stl': -1.}]))
    reporter.main(tmp_path)
    viewer = (out / 'index.html').read_text()
    assert '000_safe_arc' in viewer and 'safe_arc.mp4' in viewer
    assert 'false_safe_05' in viewer and 'Physical safe: <b>False' in viewer
    plot = out / 'trace_000_safe_arc.png'
    assert plot.stat().st_size > 1000
    first_mtime = plot.stat().st_mtime_ns
    reporter.main(tmp_path)
    assert plot.stat().st_mtime_ns == first_mtime
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['report_warnings'] == []
    assert len(summary['aggregates']) == 2
    assert (tmp_path / 'comparison.png').exists()
