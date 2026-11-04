"""Exact-source and plan-binding gates before any remote evaluation action."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import socket
import sys

import pytest

from work import operate_c2_policy_evaluation as operator
from work.prepare_c2_policy_evaluation import binding


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def ready(tmp_path, monkeypatch):
    original = Path(operator.ROOT)
    monkeypatch.setattr(operator, 'ROOT', tmp_path)
    for name in (operator.ENTRY, *operator.NEW_SOURCES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((original / name).read_bytes())
    plans = {}
    for name in ('smoke', 'lasher'):
        plan = json.loads((original /
            'reports/c2_policy_evaluation_r1_20261104/dropmoe/zero' /
            (name + '.json')).read_bytes())
        plans[name] = plan
        write(tmp_path / 'plans' / (name + '.json'), plan)
    write(tmp_path / 'reports/c2_policy_fit_r1_20261104/dropmoe/zero/'
          'completion_witness/plan.json', {'root': '/original-fit'})
    review = dict(stage='P2_C2_policy_evaluation', status='SOURCE_PASS',
        blocking_issues=0, sources=plans['lasher']['sources'],
        admitted_plan_bindings=[binding(p) for p in plans.values()],
        operational_sources={operator.ENTRY: hashlib.sha256(
            (tmp_path / operator.ENTRY).read_bytes()).hexdigest()})
    write(tmp_path / 'review.json', review)
    return tmp_path, plans, review


@pytest.mark.parametrize('action', ['deploy', 'smoke', 'full', 'inspect'])
def test_exact_review_prepares_enabled_canonical_plans_without_mutating_drafts(ready, action):
    root, plans, _ = ready
    before = {p: p.read_bytes() for p in (root / 'plans').iterdir()}
    request = operator.prepare_request(root / 'plans', root / 'review.json', action)
    assert request['source_root'] == '/original-fit'
    assert request['action'] == action
    assert request['operational_sources'][operator.ENTRY]
    assert set(request['new_sources']) == set(operator.NEW_SOURCES)
    for name, plan in request['plans'].items():
        assert plan['launch_allowed'] is True
        assert plan['review']['sha256'] == request['review_sha256']
        assert binding(plan) == binding(plans[name])
    assert before == {p: p.read_bytes() for p in before}
    assert hashlib.sha256(base64.b64decode(request['review_raw'])).hexdigest() == request['review_sha256']


@pytest.mark.parametrize('field,value', [
    ('status', 'SOURCE_FAIL'), ('blocking_issues', 1),
    ('admitted_plan_bindings', []), ('operational_sources', {}),
    ('sources', {}),
])
def test_wrong_review_fails_before_transport(ready, field, value):
    root, _, review = ready
    review[field] = value
    write(root / 'review.json', review)
    with pytest.raises(AssertionError):
        operator.prepare_request(root / 'plans', root / 'review.json', 'deploy')


def test_source_drift_rejected(ready):
    root, _, _ = ready
    with (root / operator.ENTRY).open('ab') as handle:
        handle.write(b'\n# changed\n')
    with pytest.raises(AssertionError):
        operator.prepare_request(root / 'plans', root / 'review.json', 'deploy')


def test_plan_drift_rejected(ready):
    root, plans, _ = ready
    plans['lasher']['checkpoint']['sha256'] = 'changed'
    write(root / 'plans/lasher.json', plans['lasher'])
    with pytest.raises(AssertionError):
        operator.prepare_request(root / 'plans', root / 'review.json', 'deploy')


def test_remote_payload_compiles_and_rejects_review_hash_drift(ready, monkeypatch):
    root, _, _ = ready
    request = operator.prepare_request(root / 'plans', root / 'review.json', 'deploy')
    request['review_sha256'] = 'changed'
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(request)))
    with pytest.raises(AssertionError):
        exec(compile(operator.REMOTE, '<remote>', 'exec'), {})
    assert not Path(request['plans']['lasher']['root']).exists()
