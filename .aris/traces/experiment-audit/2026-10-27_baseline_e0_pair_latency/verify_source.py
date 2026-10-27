"""Offline stdlib-only source audit; never import project or checkpoint code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
PLAN_PATH = 'reports/BASELINE_E0_PAIRED_LATENCY_PLAN_20261027.json'
plan = json.loads((ROOT / PLAN_PATH).read_bytes())
baseline = json.loads((ROOT / 'reports/CADTRACK_BASELINE_LATENCY_PLAN_20261027.json').read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


hash_rows = []
syntax_rows = []
for name, expected in plan['source_sha256'].items():
    path = ROOT / name
    actual = digest(path)
    hash_rows.append(dict(path=name, expected=expected, actual=actual, match=expected == actual))
    if path.suffix == '.py':
        row = dict(path=name)
        try:
            source = path.read_bytes()
            ast.parse(source, filename=name, feature_version=(3, 7))
            compile(source, name, 'exec', dont_inherit=True)
            row['status'] = 'PASS'
        except (SyntaxError, UnicodeError, ValueError) as error:
            row.update(status='FAIL', error=str(error))
        syntax_rows.append(row)

# Conservative static import closure: include imports in functions only for the
# entry point and the two benchmark factories actually called by the profiler.
search_roots = [ROOT / 'scripts', ROOT, ROOT / 'tools']
dynamic_function_imports = {
    'tools/profile_baseline_e0_pair.py': {'main'},
    'scripts/run_benchmark_tracking.py': {'build_model', 'build_tracker'},
}


def local_modules(module):
    pieces = module.split('.') if module else []
    for base in search_roots:
        path = base.joinpath(*pieces)
        candidate = path.with_suffix('.py')
        if candidate.is_file():
            result = [candidate]
        elif (path / '__init__.py').is_file():
            result = [path / '__init__.py']
        elif path.is_dir():
            result = []
        else:
            continue
        for length in range(1, len(pieces)):
            initializer = base.joinpath(*pieces[:length], '__init__.py')
            if initializer.is_file():
                result.append(initializer)
        return result
    return []


def selected_imports(name):
    tree = ast.parse((ROOT / name).read_bytes(), filename=name)
    selected = []

    def visit(node, inside_function=False):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in dynamic_function_imports.get(name, set()):
                return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child, inside_function)

    visit(tree)
    return selected


seen = set()
pending = ['tools/profile_baseline_e0_pair.py']
edges = []
while pending:
    name = pending.pop()
    if name in seen:
        continue
    seen.add(name)
    for node in selected_imports(name):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            if node.level:
                package = Path(name).parent.parts
                prefix = package[:len(package) - node.level + 1]
                module = '.'.join(prefix + tuple((node.module or '').split('.'))).rstrip('.')
            else:
                module = node.module or ''
            modules.append(module)
            modules.extend('.'.join(filter(None, [module, alias.name])) for alias in node.names if alias.name != '*')
        for module in modules:
            for path in local_modules(module):
                target = path.relative_to(ROOT).as_posix()
                edge = dict(source=name, line=node.lineno, module=module, target=target)
                if edge not in edges:
                    edges.append(edge)
                if target not in seen:
                    pending.append(target)

missing = sorted(seen - set(plan['source_sha256']))
metadata_names = [PLAN_PATH,
    'reports/CADTRACK_BASELINE_LATENCY_PLAN_20261027.json',
    'reports/CADTRACK_BASELINE_LATENCY_SOURCE_REVIEW_20261027.json',
    'reports/CADTRACK_BASELINE_LATENCY_RESULT_AUDIT_20261027.json',
    'reports/SERVER25_REMAINING_GRID_EXECUTION_20261027.json',
    'reports/PAIRED_LATENCY_E0_RESTORE_20261027.json']
report = dict(
    plan_sha256=digest(ROOT / PLAN_PATH),
    source_count=len(hash_rows),
    python_source_count=len(syntax_rows),
    hash_mismatches=[row for row in hash_rows if not row['match']],
    syntax_failures=[row for row in syntax_rows if row['status'] != 'PASS'],
    baseline_inherited_count=len(baseline['source_sha256']),
    baseline_inherited_changes=[name for name, expected in baseline['source_sha256'].items()
                                if plan['source_sha256'].get(name) != expected],
    new_sources=sorted(set(plan['source_sha256']) - set(baseline['source_sha256'])),
    closure_count=len(seen),
    unbound_imported_sources=[dict(path=name, sha256=digest(ROOT / name),
                                  bytes=(ROOT / name).stat().st_size) for name in missing],
    source_hashes=hash_rows,
    syntax=syntax_rows,
    static_import_edges=edges,
    metadata_sha256={name: digest(ROOT / name) for name in metadata_names},
    local_manifest_present=(ROOT / 'data/manifest.json').is_file(),
    limitations='Static import graph includes branch-local e1 factory imports; no module was imported, no neural or checkpoint code executed. External installed packages and remote presence were not inspected.')
(TRACE / 'deterministic_source_checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in report.items()
                  if key not in ('source_hashes', 'syntax', 'static_import_edges')}, indent=2))
