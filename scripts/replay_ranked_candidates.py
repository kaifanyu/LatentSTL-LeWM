"""Replay saved offline candidates; this does not run CEM or bypass the MPC gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from latent_stl_lewm.simulator import MugPreservationEnv


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def video(path, frames, fps):
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=9,
                           macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
        for frame in frames:
            writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-gpu', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selections, full_scores, requested = [], {}, defaultdict(list)
    for seed in range(3):
        directory = args.campaign / f'seed_{seed}' / 'evaluation'
        full = {row['id']: row for row in json.loads((directory/'level0_trajectories.json').read_text())}
        banks = json.loads((directory/'level2_candidates.json').read_text())
        best_pred = max((row for bank in banks.values() for row in bank), key=lambda row: row['predicted']['stl'])
        best_actual = max(full.values(), key=lambda row: row['scores']['stl'])
        full_scores[seed] = full
        for role, chosen in [('highest_predicted_8step', best_pred), ('highest_actual_full_trace', best_actual)]:
            identifier = chosen['id']
            row = {'seed': seed, 'role': role, 'trajectory_id': identifier,
                   'policy': full[identifier]['metadata']['policy'],
                   'bank_id': full[identifier]['metadata']['bank_id'],
                   'full_actual_stl': full[identifier]['scores']['stl'],
                   'physical_labels': full[identifier]['labels']}
            if role == 'highest_predicted_8step':
                row.update(predicted_stl_8step=chosen['predicted']['stl'], actual_stl_8step=chosen['actual']['stl'])
            selections.append(row)
            requested[identifier].append(f'seed {seed}: {role}')
    # A successful control from the exact same start as seed0's selected path.
    chosen_bank = selections[0]['bank_id']
    control = next(row for row in full_scores[0].values()
                   if row['metadata']['bank_id'] == chosen_bank and row['metadata']['policy'] == 'safe_arc')
    requested[control['id']].append('same-start successful scripted control')
    result = {'source_campaign': str(args.campaign.resolve()), 'dataset_sha256': digest(args.dataset),
              'scope': 'New simulator replay of recorded scripted candidates selected from existing offline evaluation. No CEM/SVGD optimization or MPC replanning.',
              'ranking_scope': 'Highest recorded score within each seed evaluation, across its 32 test trajectories; raw scores are not comparable across seeds.',
              'selections': selections, 'trajectories': {}}
    replay_frames, recorded_frames = {}, {}
    with h5py.File(args.dataset, 'r') as data:
        metadata = json.loads(data.attrs['metadata_json'])
        cfg = metadata['simulator_config'].copy(); cfg['render_gpu'] = args.render_gpu
        fps = int(metadata['frame_freq_hz'])
        Image.fromarray(data['references/goal_rgb'][:]).save(args.output/'goal.png')
        Image.fromarray(data['references/safe_rgb'][:]).save(args.output/'safe.png')
        env = MugPreservationEnv(cfg)
        try:
            for identifier, roles in requested.items():
                episode = data['trajectories'][identifier]
                bank = str(episode.attrs['bank_id'])
                snapshot = {key: value[()] for key, value in data['start_snapshots'][bank].items()}
                env.reset(int(episode.attrs['scene_seed']))
                frames = [env.restore(snapshot)]; diagnostics = [env.diagnostics()]
                actions = episode['action'][:]
                for action in actions:
                    frame, _, _ = env.step(action)
                    frames.append(frame); diagnostics.append(env.diagnostics())
                replay, original = np.stack(frames), episode['rgb'][:]
                replay_frames[identifier], recorded_frames[identifier] = replay, original
                video(args.output/f'{identifier}_replay.mp4', replay, fps)
                video(args.output/f'{identifier}_recorded.mp4', original, fps)
                np.save(args.output/f'{identifier}_actions.npy', actions)
                eef = np.stack([x['eef_pos'] for x in diagnostics])
                original_diag = episode['diagnostics']
                entry = {'roles': roles, 'policy': str(episode.attrs['policy']), 'bank_id': bank,
                         'steps': len(actions), 'fps': fps,
                         'rgb_exact_match': bool(np.array_equal(replay, original)),
                         'max_rgb_absolute_difference': int(np.abs(replay.astype(np.int16)-original.astype(np.int16)).max()),
                         'max_eef_replay_error_m': float(np.abs(eef-original_diag['eef_pos'][:]).max()),
                         'physical_safe': all(bool(d['safe']) for d in diagnostics),
                         'physical_reached': any(bool(d['reach']) for d in diagnostics),
                         'final_goal_error_m': float(diagnostics[-1]['goal_error']),
                         'maximum_mug_displacement_m': max(float(d['object_displacement']) for d in diagnostics),
                         'maximum_mug_rotation_deg': max(float(d['object_rotation_deg']) for d in diagnostics),
                         'actions_sha256': hashlib.sha256(actions.tobytes()).hexdigest()}
                result['trajectories'][identifier] = entry
                print(json.dumps({'trajectory_id': identifier, **entry}), flush=True)
        finally:
            env.close()

    first_pred = next(x for x in selections if x['seed'] == 0 and x['role'] == 'highest_predicted_8step')
    first_full = next(x for x in selections if x['seed'] == 0 and x['role'] == 'highest_actual_full_trace')
    panels = [(first_full['trajectory_id'], 'Highest full-trace score', 'NO-OP: safe, no reach'),
              (first_pred['trajectory_id'], 'Highest predicted 8-step score', 'STRAIGHT: later disturbs mug'),
              (control['id'], 'Successful same-start control', 'ARC: safe reach; lower STL score')]
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 14)
    comparison = []
    for t in range(len(recorded_frames[panels[0][0]])):
        canvas = Image.new('RGB', (1080, 496), '#101827'); draw = ImageDraw.Draw(canvas)
        draw.text((16, 8), 'Seed 0 | saved simulator trajectories | no MPC was executed', fill='white', font=font)
        for col, (identifier, title, caption) in enumerate(panels):
            x = col*360
            draw.text((x+12, 40), title, fill='#b9d4ff', font=font)
            # Enlarge the recorded 128px camera image for visibility, without new rendering geometry.
            image = Image.fromarray(recorded_frames[identifier][t]).resize((336,336), Image.Resampling.NEAREST)
            canvas.paste(image, (x+12, 68))
            draw.text((x+12, 412), caption, fill='white', font=small)
            score = full_scores[0][identifier]['scores']['stl']
            draw.text((x+12, 438), f'Actual full-trace rho: {score:+.3f}', fill='#fbbf24', font=small)
            draw.text((x+12, 462), f'Step {t:02d}/96 | {t/fps:.2f} s', fill='white', font=small)
        if t<=8:
            draw.text((600, 8), 'Within the scored 8-step prefix', fill='#8bf0bc', font=small)
        else:
            draw.text((600, 8), 'Beyond the scored 8-step prefix', fill='#fbbf24', font=small)
        comparison.append(np.asarray(canvas))
    video(args.output/'seed0_comparison.mp4', comparison, fps)
    Image.fromarray(comparison[48]).save(args.output/'comparison_preview.png')
    (args.output/'replay_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    cards = []
    for row in selections:
        identifier = row['trajectory_id']; d=result['trajectories'][identifier]
        role = row['role'].replace('_', ' ')
        predicted = (f'Predicted 8-step STL: {row["predicted_stl_8step"]:.3f}. '
                     if 'predicted_stl_8step' in row else '')
        cards.append(f'<article><h3>Seed {row["seed"]}: {html.escape(role)}</h3>'
                     f'<p>Trajectory {identifier}; {html.escape(row["policy"])}. '
                     f'{predicted}Full actual STL: {row["full_actual_stl"]:.3f}. '
                     f'Safe={d["physical_safe"]}, reached={d["physical_reached"]}.</p>'
                     f'<video controls loop preload="metadata" src="{identifier}_replay.mp4"></video></article>')
    page='''<!doctype html><html><head><meta charset="utf-8"><title>LeWM recorded candidate replays</title>
<style>body{font:16px system-ui;background:#101827;color:#eef2ff;max-width:1150px;margin:32px auto;padding:0 20px}video{width:100%;max-width:1080px}article video{max-width:384px;image-rendering:pixelated}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}article{padding:16px;background:#1b2639}a{color:#99bfff}</style></head><body>
<h1>Recorded candidates and verified simulator replays</h1>
<p>The campaign trained LeWM and ranked scripted candidates offline. CEM/SVGD and closed-loop MPC were not executed.
Predicted ranking covered only 8 actions (0.4s); the videos show all 96 saved actions (4.8s). All reported STL scores are negative.</p>
<h2>Same start, three different outcomes (seed 0)</h2><video controls loop preload="metadata" poster="comparison_preview.png" src="seed0_comparison.mp4"></video>
<p>The comparison uses original simulator frames. Individual videos below are freshly replayed saved actions; bytewise RGB and physical replay agreement are recorded in the manifest.</p>
<p><a href="replay_manifest.json">Selection and replay verification</a></p><div class="grid">'''+''.join(cards)+'</div></body></html>'
    (args.output/'index.html').write_text(page)
    print(str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
