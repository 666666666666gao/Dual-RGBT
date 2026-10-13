"""CPU analysis of saved, paired A7 dev crops; no new neural execution."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows):
    return dict(samples=len(rows), mean_loss=mean(float(r['localization_loss']) for r in rows),
                mean_iou=mean(float(r['sample_iou']) for r in rows),
                iou50_count=sum(float(r['sample_iou']) >= .5 for r in rows),
                modality_counts=dict(Counter(r['modality'] for r in rows)),
                action_counts=dict(Counter(r['action'] for r in rows)),
                mean_heavy_tokens=mean(385 + (0 if r['action'] == '0' else
                                             385 if r['action'] == '4' else 193) for r in rows))


def main():
    source = ROOT / 'reports/stage_a_full/dev_samples_epoch07.csv'
    rows = list(csv.DictReader(source.open()))
    paired = {(int(r['sample_id']), int(r['modality']), int(r['action'])): r for r in rows}
    assert len(rows) == len(paired) == 1000
    assert set(paired) == {(s, m, a) for s in range(100) for m in range(2) for a in range(5)}
    for sample in range(100):
        identity = {(paired[sample, m, a]['sequence'], paired[sample, m, a]['frame'])
                    for m in range(2) for a in range(5)}
        assert len(identity) == 1
    policies = {f'fixed_{m}_{a}': [paired[s, m, a] for s in range(100)]
                for m in range(2) for a in range(5)}
    loss = lambda row: float(row['localization_loss'])
    for modality in range(2):
        policies[f'GT_diagnostic_fixed_{modality}_best_action'] = [
            min((paired[s, modality, a] for a in range(5)), key=loss) for s in range(100)]
    best_main = [min((paired[s, m, 0] for m in range(2)), key=loss) for s in range(100)]
    policies['GT_diagnostic_best_main_STOP'] = best_main
    policies['GT_diagnostic_best_main_then_action'] = [
        min((paired[s, int(main['modality']), a] for a in range(5)), key=loss)
        for s, main in enumerate(best_main)]
    policies['GT_diagnostic_joint_best_path'] = [
        min((paired[s, m, a] for m in range(2) for a in range(5)), key=loss) for s in range(100)]
    strata = []
    for modality in range(2):
        for label, lower, upper in [('low_iou', 0., .5), ('middle_iou', .5, .8), ('high_iou', .8, 1.01)]:
            samples = [s for s in range(100) if lower <= float(paired[s, modality, 0]['sample_iou']) < upper]
            gains = [loss(paired[s, modality, 0]) - loss(paired[s, modality, 1]) for s in samples]
            strata.append(dict(modality=modality, stratum=label, samples=len(samples),
                               mean_primary_gain=mean(gains), negative_count=sum(g < 0 for g in gains)))
    comparisons = []
    for modality in range(2):
        for action in range(1, 4):
            difference = [float(paired[s, modality, action]['sample_iou']) -
                          float(paired[s, modality, 4]['sample_iou']) for s in range(100)]
            comparisons.append(dict(modality=modality, action=action,
                                    mean_iou_delta_local_minus_expand=mean(difference),
                                    higher_iou_count=sum(d > 0 for d in difference),
                                    lower_iou_count=sum(d < 0 for d in difference)))
    coverage = []
    for modality in range(2):
        for label, predicate in [('high_coverage', lambda c: c >= .9),
                                 ('other_coverage', lambda c: c < .9)]:
            selected = [s for s in range(100) if predicate(float(paired[s, modality, 1]['gt_coverage']))]
            gains = [loss(paired[s, modality, 0]) - loss(paired[s, modality, 1]) for s in selected]
            coverage.append(dict(modality=modality, stratum=label, samples=len(selected),
                                 mean_gain=mean(gains), negative_count=sum(g < 0 for g in gains)))
    report = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  paired_samples=100, distinct_sequences=len({r['sequence'] for r in rows}),
                  policies={name: summarize(values) for name, values in policies.items()},
                  primary_gain_by_initial_iou=strata, primary_gain_by_gt_coverage=coverage,
                  local_vs_expand=comparisons,
                  scope='Saved GT-centered dev-crop diagnosis only. GT policies are not deployable, '
                        'not tracking-trajectory bounds, not formal PR/NPR/SR, and token counts are not latency.')
    output = ROOT / 'reports/stage_a_full/opportunity_analysis.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
