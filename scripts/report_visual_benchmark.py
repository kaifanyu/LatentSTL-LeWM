"""Refresh the visual-STL campaign's local HTML report from completed artifacts."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np


def atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text); tmp.replace(path)


def main(root):
    root = Path(root)
    protocol_path = root / "protocol.json"
    planned = json.loads(protocol_path.read_text()).get("fresh_trajectories", 864) if protocol_path.exists() else 864
    records = []
    for path in sorted((root / "evaluations").glob("*/*/report.json")):
        report = json.loads(path.read_text())
        for op, r in report["operating_points"].items():
            records.append({"shard": path.parent.parent.name, "method": report["method"],
                            "operating_point": op, "report": str(path.relative_to(root)), **r})
    aggregates = []
    for method in ("global", "spatial", "pixels", "red_pixels"):
        for op in ("balanced", "false_safe_05"):
            rows = [r for r in records if r["method"] == method and r["operating_point"] == op]
            if not rows:
                continue
            def ratio(num, den):
                return num / den if den else None
            count = sum(r["episodes"] for r in rows)
            safe_negative = sum(r["frame_safe"]["negatives"] for r in rows)
            safe_positive = sum(r["frame_safe"]["positives"] for r in rows)
            ranks = {}
            for obj in ("terminal", "reach_max", "weighted_penalty", "stl", "reach_and_hold"):
                ranks[obj] = {key: sum(r["offline_candidate_ranking"][obj][key] for r in rows)
                              for key in ("banks", "safe_reach", "unsafe", "safe_hold")}
            aggregates.append({"method": method, "operating_point": op, "episodes": count,
                "frame_false_safe": ratio(sum(r["frame_safe"]["false_safe_count"] for r in rows), safe_negative),
                "frame_false_unsafe": ratio(sum(r["frame_safe"]["false_unsafe_count"] for r in rows), safe_positive),
                "stl_agreement": ratio(sum(r["trajectory_stl"]["sign_agreement"]*r["episodes"] for r in rows), count),
                "missed_violations": sum(r["missed_violations"] for r in rows),
                "unsafe_episodes": sum(r["unsafe_episodes"] for r in rows), "ranking": ranks})
    collected = []
    for path in sorted((root / "data").glob("*/collection.json")):
        data = json.loads(path.read_text())
        collected.append({"shard": path.parent.name, "episodes": len(data["episodes"]),
                          "seconds": data["seconds"]})
    summary = {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "fresh_trajectories_completed": sum(r["episodes"] for r in collected),
               "completed_shards": collected, "aggregates": aggregates, "by_shard": records,
               "scope": "Observed visual STL and recorded-action forecast diagnostics; collection scripts are not planned actions"}
    atomic(root / "summary.json", json.dumps(summary, indent=2) + "\n")
    def pct(v):
        return "—" if v is None else f"{v:.1%}"
    table = ""
    for r in aggregates:
        rank = r["ranking"]["stl"]
        table += (f"<tr><td>{r['method']}</td><td>{r['operating_point']}</td><td>{r['episodes']}</td>"
                  f"<td>{pct(r['frame_false_safe'])}</td><td>{pct(r['frame_false_unsafe'])}</td>"
                  f"<td>{pct(r['stl_agreement'])}</td><td>{r['missed_violations']}/{r['unsafe_episodes']}</td>"
                  f"<td>{rank['safe_reach']}/{rank['banks']}</td></tr>")
    details = "".join(f"<li><a href='{html.escape(r['report'])}'>{html.escape(r['shard'])}: {r['method']} / {r['operating_point']}</a></li>" for r in records)
    videos = "".join(f"<details><summary>{html.escape(p.parent.name)} / {html.escape(p.stem)}</summary>"
                     f"<video controls preload='none' width='384' src='{html.escape(str(p.relative_to(root)))}'></video></details>"
                     for p in sorted((root / "data").glob("*/*.mp4")))
    historical = "".join(f"<li><a href='{p.relative_to(root)}'>{p.parent.name}</a></li>"
                         for p in sorted((root / "calibration").glob("*/report.json")))
    forecasts = "".join(f"<li><a href='{p.relative_to(root)}'>{p.parent.name}</a></li>"
                        for p in sorted((root / "forecasts").glob("*/report.json")))
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='60'>
<title>Visual STL benchmark</title><style>body{{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 20px;background:#f7f8fa;color:#182334}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}details{{padding:8px}}a{{color:#175dad}}.note{{padding:16px;background:#eaf1ff}}</style></head><body>
<h1>Visual and latent STL benchmark</h1><p>Updated {summary['updated_at_utc']}. {summary['fresh_trajectories_completed']} / {planned} fresh trajectories collected.</p>
<p class='note'>This campaign evaluates image predicates, temporal logic, and offline candidate ranking. Trajectories are generated by scripted collection controllers. It does not claim CEM-generated actions or closed-loop planning success. The existing LeWM forecasts global latents only.</p>
<p><a href='status.json'>Live process status</a> · <a href='protocol.json'>Frozen protocol</a> · <a href='summary.json'>Full metrics</a> · <a href='frozen_calibration.json'>Calibration hashes</a></p>
<h2>Fresh held-out data</h2><p>Two operating points are frozen using historical validation data. False-safe is the fraction of physically disturbed frames accepted as safe. Misses are whole unsafe episodes with no alarm after disturbance. Confidence intervals are in individual shard reports and resample initial-state banks.</p>
<table><tr><th>Representation</th><th>Calibration</th><th>Episodes</th><th>False safe ↓</th><th>False unsafe ↓</th><th>STL agreement ↑</th><th>Missed disturbances ↓</th><th>Offline STL selected safe reach</th></tr>{table}</table>
<h2>Historical diagnostic</h2><p>Previously inspected test split, reported separately from new held-out scenes.</p><ul>{historical}</ul>
<h2>Forecast audits</h2><ul>{forecasts}</ul><h2>Detailed scenario results</h2><ul>{details}</ul>
<h2>Representative simulated trajectories</h2><p>Each video is the first bank's actual simulator rollout; subsequent banks remain available in the HDF5 shards.</p>{videos}</body></html>"""
    atomic(root / "index.html", page)
    if aggregates:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        selected = [r for r in aggregates if r["operating_point"] == "balanced"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric, title in zip(axes, ("frame_false_safe", "frame_false_unsafe", "stl_agreement"),
                                    ("Unsafe frames accepted ↓", "Safe frames rejected ↓", "STL / physical agreement ↑")):
            vals = [r[metric] if r[metric] is not None else np.nan for r in selected]
            ax.bar([r["method"] for r in selected], vals)
            ax.set_ylim(0, 1); ax.set_title(title); ax.tick_params(axis="x", rotation=25)
        fig.suptitle("Fresh trajectories — validation balanced-accuracy thresholds")
        fig.tight_layout(); fig.savefig(root / "comparison.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("root", type=Path)
    main(p.parse_args().root)
