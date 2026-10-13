"""CPU analysis/plots of archived complete B supervision and native checks."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / 'reports/full_pipeline/stage_b_refreshed_20261013/policy_steps.csv').open(newline='')))
fields = ('loss', 'quality_loss', 'gain_loss', 'preview_loss', 'negative_fraction')
epochs = []
for epoch in range(1, 6):
    group = [r for r in rows if int(r['epoch']) == epoch]
    epochs.append(dict(epoch=epoch, steps=len(group),
                       **{k: statistics.mean(float(r[k]) for r in group) for k in fields}))
residual = max(abs(float(r['loss']) - float(r['quality_loss']) - float(r['gain_loss']) -
                   .1 * float(r['preview_loss'])) for r in rows)
assert residual < 1e-6
checks = {}
for label, filename in [('full_b', 'reports/b_complete_check.json'),
                         ('b_sanity', 'reports/b_a7_batch8_check.json')]:
    frames = json.loads((root / filename).read_text())['frames']
    checks[label] = dict(frames=len(frames),
                         modality_counts=dict(Counter(r['modality'] for r in frames)),
                         action_counts=dict(Counter(r['action'] for r in frames)),
                         mean_heavy_tokens=statistics.mean(sum(r['heavy_token_trace']) for r in frames))
result = dict(status='PASS', epoch_means=epochs, maximum_loss_formula_residual=residual,
              completed_native_prefix_routes=checks,
              scope='Completed B supervision and 32-frame training-prefix execution diagnostics only. '
                    'No C progress, GT tracking accuracy, formal speed, or independent module evidence.')
(root / 'reports/stage_b_training_analysis.json').write_text(json.dumps(result, indent=2))
fig, axes = plt.subplots(2, 2, figsize=(9, 5.3), constrained_layout=True)
for ax, field in zip(axes.flat, fields[:4]):
    x, y = [], []
    for start in range(0, len(rows), 50):
        group = rows[start:start + 50]
        x.append((start + len(group) / 2) / 1250)
        y.append(statistics.mean(float(r[field]) for r in group))
    ax.plot(x, y, color='#245d8d', linewidth=1.15, label='50-update block mean')
    ax.plot([r['epoch'] - .5 for r in epochs], [r[field] for r in epochs],
            marker='o', color='#b56022', label='Epoch mean')
    ax.set_title(field.replace('_', ' '), fontsize=11)
    ax.set_xlabel('Epoch progress')
    ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8)
fig.suptitle('Completed Stage B training supervision (6,250 updates)\n'
              'Training losses only; no validation/test tracking metrics', fontsize=12)
fig.savefig(root / 'reports/stage_b_training_losses.svg')
fig.savefig(root / 'reports/stage_b_training_losses.png', dpi=150)
print(json.dumps(result, indent=2))
