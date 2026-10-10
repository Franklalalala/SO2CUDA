#!/usr/bin/env python3
"""Generate the Chinese benchmark section of the README and public, allowlisted JSON evidence.

Inputs are the session summaries of one H200 measurement session: every operator
configuration with all implementations measured in one process, and every
whole-model column measured with the 3/10 protocol. Only allowlisted fields reach
the public JSON files; diagnostics are scrubbed of paths and identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


BEGIN = '<!-- SO2CUDA_BENCHMARKS_BEGIN -->'
END = '<!-- SO2CUDA_BENCHMARKS_END -->'
CONTEXT_BEGIN = '<!-- SO2CUDA_MODEL_CONTEXT_BEGIN -->'
CONTEXT_END = '<!-- SO2CUDA_MODEL_CONTEXT_END -->'
ENTRIES_BEGIN = '<!-- SO2CUDA_ENTRY_TIMINGS_BEGIN -->'
ENTRIES_END = '<!-- SO2CUDA_ENTRY_TIMINGS_END -->'
LABELS = {'so2cuda': 'SO2CUDA', 'naive': '我们的纯 PyTorch', 'eqv3': 'EquiformerV3 原版',
          'eqv3+compile': 'EquiformerV3 + compile', 'cueq': 'cuEquivariance'}
METHODS = {'naive', 'uniform_1d', 'fused_tp', 'indexed_linear'}
DESCRIPTORS = {'escn_tp', 'escn_tp_compact'}
SO2CUDA_CANDIDATES = {'dense_pairs', 'dense_pairs_grouped', 'true_dense_pairs'}
TABLE_ENTRY = 'true_dense_pairs'  # the general entry shown in the README tables
ENTRY_LABELS = (('true_dense_pairs', '`true_dense_pairs`'), ('dense_pairs', '`dense_pairs`'),
                ('dense_pairs_grouped', '`dense_pairs`（grouped）'), ('activation', '`activation_forward`（单专家）'))
MODEL_COLUMNS = ('naive', 'cueq', 'so2cuda')
CLARIFICATION = ('“我们的纯 PyTorch”是我们按 DeePTB 上游 `SO2_Linear` 与 UMA MoLE 写法自己实现的'
    '（[examples/naive_baseline.py](examples/naive_baseline.py)）。朴素 PyTorch SO(2) 基线采用 '
    'EquiformerV3 原版 `SO3Rotation` ＋ `SO2Linear`（eager）；模型层的通道布局不适用时标 N/A。')
CUEQ_MODEL_FILES = ('cueq_baseline.py', 'operator_cueq.py', 'deeptb_speed_test.py')


def public_reason(value):
    """Preserve diagnostics while removing private path and host identities."""
    if not isinstance(value, str):
        raise ValueError('Diagnostic reason must be text')
    value = re.sub(r'(?:/[A-Za-z0-9_.+~-]+){2,}', '<path>', value)
    value = re.sub(r'\b(?:[A-Za-z]:\\)[^\s\"\']+', '<path>', value)
    value = re.sub(r'\b[^\s@]+@[^\s@]+\b', '<identity>', value)
    value = re.sub(r'GPU-[A-Za-z0-9-]+', '<gpu>', value)
    return value


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected finite numeric evidence, got {value!r}')
    return value


def commits(source):
    found = {}
    aliases = {'so2cuda': 'SO2CUDA', 'deeptb': 'DeePTB', 'equiformerv3': 'EquiformerV3'}
    for key, sha in source.get('commits', {}).items():
        name = aliases.get(key.lower().replace('_', ''))
        if name and isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{40}', sha):
            found[name] = sha
    if set(found) != {'SO2CUDA', 'DeePTB', 'EquiformerV3'}:
        raise ValueError('Source commit provenance is incomplete')
    return found


def measurement(row, scope):
    value = row.get(scope, {})
    if value:
        result = {key: number(value[key]) for key in ('median_ms', 'q1_ms', 'q3_ms')}
        if 'samples_ms' in value:
            result['samples_ms'] = [number(x) for x in value['samples_ms']]
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            if key in value:
                result[key] = number(value[key])
    else:
        values = row.get('timing_statistics_ms', {}).get(scope)
        if values is None:
            raise ValueError(f'Missing recorded {scope} quartiles')
        result = {'median_ms': number(values['median']), 'q1_ms': number(values['q1']),
                  'q3_ms': number(values['q3'])}
        samples = row.get(scope + '_samples_ms')
        if scope == 'forward_backward' and 'forward_samples_ms' in row and 'backward_samples_ms' in row:
            samples = [a + b for a, b in zip(row['forward_samples_ms'], row['backward_samples_ms'])]
        if samples is not None:
            result['samples_ms'] = [number(x) for x in samples]
        for key in ('peak_allocated', 'peak_reserved'):
            if key + '_gib' in row:
                result[key + '_bytes'] = number(row[key + '_gib']) * 2 ** 30
    if result['median_ms'] <= 0 or not (0 <= result['q1_ms'] <= result['median_ms'] <= result['q3_ms']):
        raise ValueError('Invalid recorded timing median or quartiles')
    if result.get('samples_ms'):
        samples = result['samples_ms']
        q1, _, q3 = statistics.quantiles(samples, n=4, method='inclusive') if len(samples) > 1 else [samples[0]] * 3
        for key, expected in (('median_ms', statistics.median(samples)), ('q1_ms', q1), ('q3_ms', q3)):
            if not math.isclose(result[key], expected, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError('Recorded timing summary does not reproduce from samples')
    return result


def metrics(value):
    """Keep numerical comparison evidence without arbitrary source or parameter names."""
    rows = []
    quantities = ('output', 'input_gradient', 'weight_gradients', 'parameter_gradients',
                  'canonical_weight_gradients', 'training_outputs', 'inference_outputs', 'equivariance')

    def visit(node, quantity='unspecified'):
        if isinstance(node, dict):
            if 'max_abs' in node and ('relative_l2' in node or 'max_relative_l2' in node):
                row = {'quantity': quantity, 'max_abs': number(node['max_abs']),
                       'relative_l2': number(node.get('relative_l2', node.get('max_relative_l2')))}
                for key in ('passed', 'finite'):
                    if key in node:
                        row[key] = bool(node[key])
                for key in ('rms_error', 'reference_max_abs', 'reference_rms', 'atol', 'rtol'):
                    if key in node:
                        row[key] = number(node[key])
                rows.append(row)
            for key, child in node.items():
                visit(child, key if key in quantities else quantity)
        elif isinstance(node, list):
            for child in node:
                visit(child, quantity)

    visit(value)
    if not rows:
        raise ValueError('Equivalence evidence contains no numerical metrics')
    return {'metric_count': len(rows),
            'all_passed': all(row.get('passed', True) and row.get('finite', True) for row in rows),
            'max_abs': max(row['max_abs'] for row in rows),
            'max_relative_l2': max(row['relative_l2'] for row in rows), 'metrics': rows}


def implementation(row):
    status = row.get('status')
    if status not in ('passed', 'oom', 'N/A', 'na', 'unavailable', 'failed_equivalence'):
        raise ValueError(f'Unknown implementation status: {status!r}')
    result = {'status': 'N/A' if status == 'na' else status}
    metadata = row.get('metadata', {})
    for key in ('descriptor', 'method', 'rotation', 'version', 'forward_mode', 'api', 'candidate'):
        value = row.get(key, metadata.get(key))
        allowed = ((key == 'descriptor' and value in DESCRIPTORS) or
                   (key == 'method' and value in METHODS) or
                   (key == 'rotation' and value in ('pytorch', 'cueq')) or
                   (key == 'version' and isinstance(value, str) and
                    re.fullmatch(r'\d+(?:\.\d+){1,3}[A-Za-z0-9+.-]*', value)) or
                   (key == 'candidate' and value in SO2CUDA_CANDIDATES) or
                   (key == 'forward_mode' and value in ('default', 'indexed_sandwich_multi',
                       'indexed_sandwich_multi_grouped', 'scalar')) or
                   (key == 'api' and value in ('so2_cuda_ops.deeptb.dense_pairs',
                       'so2_cuda_ops.deeptb.true_dense_pairs', 'so2_cuda_ops.deeptb.activation_forward')))
        if allowed:
            result[key] = value
    if status == 'passed':
        result['forward'] = measurement(row, 'forward')
        result['forward_backward'] = measurement(row, 'forward_backward')
    if 'feature_layout' in row:
        result['feature_layout'] = public_reason(row['feature_layout'])
    if 'memory_scope' in row:
        scope = row['memory_scope']
        result['memory_scope'] = {key: public_reason(scope[key]) for key in
            ('canonical_data', 'other_implementations', 'reserved_note') if key in scope}
        result['memory_scope']['includes'] = [public_reason(value) for value in scope.get('includes', [])]
        if 'allocated_baseline_bytes' in scope:
            result['memory_scope']['allocated_baseline_bytes'] = number(scope['allocated_baseline_bytes'])
    if row.get('shared_wigner'):
        result['shared_wigner'] = {key: number(row['shared_wigner'][key]) for key in
            ('expected_model_forwards', 'geometry_builds', 'cache_hits', 'so2_layers')
            if key in row['shared_wigner']}
        result['shared_wigner']['passed'] = row['shared_wigner'].get('passed') is True
    if row.get('route_coverage'):
        coverage = row['route_coverage']
        result['route_coverage'] = {'public_route': public_reason(coverage['public_route'])}
        for key in ('audited_successes', 'all_forward_successes', 'expected_layer_forwards'):
            result['route_coverage'][key] = number(coverage[key])
    if 'reason' in row:
        result['reason'] = public_reason(row['reason'])
    elif row.get('error'):
        result['reason'] = public_reason(str(row['error']))
    if row.get('error_type'):
        result['error_type'] = public_reason(row['error_type'])
    if row.get('selection'):
        result['selection'] = public_reason(row['selection'])
    for key in ('compilation_and_first_call_seconds', 'construction_seconds',
                'compile_and_first_training_call_seconds', 'compile_and_first_inference_call_seconds'):
        if key in row:
            result[key] = number(row[key])
    if 'equivalence_vs_eager' in row:
        result['equivalence_vs_eager'] = metrics(row['equivalence_vs_eager'])
    if 'equivalence' in row:
        result['equivalence'] = metrics(row['equivalence'])
    if row.get('cueq_execution'):
        layers = []
        for layer in row['cueq_execution'].values():
            entry = {'descriptor': layer['descriptor'], 'method': layer['method'], 'rotation': layer['rotation']}
            for key in ('rotation_detail', 'model_feature_layout', 'descriptor_feature_layout', 'wigner_layout'):
                if key in layer:
                    entry[key] = public_reason(layer[key])
            if entry['method'] not in METHODS or entry['descriptor'] not in DESCRIPTORS or entry['rotation'] != 'pytorch':
                raise ValueError('Unknown model descriptor method')
            layers.append(entry)
        result['cueq_layers'] = layers
    return result


def require_exclusive(proof):
    if not isinstance(proof, list) or len(proof) < 2 or any(row.get('exclusive') is not True for row in proof):
        raise ValueError('Missing before/after exclusive GPU evidence')


def gpu_ids(proof):
    return {row.get('uuid') for row in proof if row.get('uuid')}


def so2cuda_choices(source):
    rows = source.get('so2cuda_alternatives', [])
    if {row.get('candidate') for row in rows} != SO2CUDA_CANDIDATES or len(rows) != len(SO2CUDA_CANDIDATES):
        raise ValueError('Every SO2CUDA public API candidate must be recorded once')
    passed = [row for row in rows if row.get('status') == 'passed']
    chosen = implementation(source['implementations']['so2cuda'])
    if not passed:
        if chosen['status'] != 'oom' or not any(row.get('status') == 'oom' for row in rows):
            raise ValueError('SO2CUDA has no numerically qualified timed candidate or recorded OOM')
    else:
        winner = min(passed, key=lambda row: measurement(row, 'forward_backward')['median_ms'])
        if chosen.get('candidate') != winner['candidate'] or chosen['forward_backward'] != measurement(winner, 'forward_backward'):
            raise ValueError('SO2CUDA column does not reuse the fastest qualified candidate')
    return rows


def cueq_choice(source):
    """The cuEquivariance column is one fixed descriptor/method/rotation, timed after its own check."""
    rows = source.get('cueq_alternatives', [])
    if len(rows) != 1:
        raise ValueError('Exactly one fixed cuEquivariance combination must be recorded')
    row = rows[0]
    if (row.get('descriptor') not in DESCRIPTORS or row.get('method') not in METHODS
            or row.get('rotation') not in ('pytorch', 'cueq')):
        raise ValueError('Unknown cuEquivariance combination')
    choice = implementation(source['implementations']['cueq'])
    if row.get('status') == 'passed':
        if not row.get('equivalence', {}).get('passed'):
            raise ValueError('cuEquivariance timing lacks its passed equivalence check')
        if (any(choice.get(key) != row.get(key) for key in ('descriptor', 'method', 'rotation'))
                or choice['forward_backward'] != measurement(row, 'forward_backward')):
            raise ValueError('cuEquivariance column does not reuse the recorded fixed combination')
    return rows


def operator_evidence(raw):
    if raw.get('schema') != 'h200-operator-session-v1' or raw.get('status') != 'completed':
        raise ValueError('Operator session summary has not completed')
    source = raw.get('payload_source', {})
    if source.get('timing_contract') != 'native-layout-v2' or raw.get('matrix', {}).get('timing_contract') != 'native-layout-v2':
        raise ValueError('Public timings require the native-layout-v2 measurement contract')
    expected = {case['id'] for case in raw['matrix']['cases'] if case['kind'] == 'operator'}
    received = [case.get('id') for case in raw['cases']]
    if len(set(received)) != len(received) or set(received) != expected:
        raise ValueError('Operator session does not contain every matrix case exactly once')
    cases, devices, checks = [], set(), []
    for wrapper in raw['cases']:
        result = wrapper.get('result')
        if wrapper.get('status') != 'completed' or result is None or result.get('status') != 'completed':
            raise ValueError('Operator case is incomplete or failed')
        if result.get('timing_contract') != 'native-layout-v2':
            raise ValueError('Operator case used another timing boundary')
        if 'H200' not in result.get('gpu', ''):
            raise ValueError('Public operator timings must be measured on H200')
        if result.get('precision') != 'strict FP32; TF32 disabled':
            raise ValueError('Operator precision evidence is missing')
        configuration = wrapper['configuration']
        config = {key: configuration[key] for key in ('lmax', 'mmax', 'channels', 'edges') if key in configuration}
        for key in ('irreps_in', 'irreps_out'):
            value = result.get('config', {}).get(key)
            if value is None or not re.fullmatch(r'[0-9xoe+ ]+', value):
                raise ValueError('Invalid public irreps representation')
            config[key] = value
        config.setdefault('mmax', result['config']['mmax'])
        config.setdefault('edges', result['config']['edges'])
        groups = configuration.get('grids', [])
        if not groups or any(group not in ('A', 'B', 'C', 'D') for group in groups):
            raise ValueError('Operator grid memberships are missing')
        row = {'config': config, 'grids': groups,
               'warmup': number(result['warmup']), 'iterations': number(result['iterations']),
               'implementations': {name: implementation(value) for name, value in result['implementations'].items()
                                   if name in LABELS}}
        if row['warmup'] < 5 or row['iterations'] < 20:
            raise ValueError('Require at least five warmups and twenty operator timing samples')
        missing = {'naive', 'so2cuda', 'eqv3', 'cueq'} - set(row['implementations'])
        if 'C' not in groups:
            missing |= {'eqv3+compile'} - set(row['implementations'])
        if missing:
            raise ValueError('Operator case lacks columns: ' + ', '.join(sorted(missing)))
        so2_rows = so2cuda_choices(result)
        cueq_rows = cueq_choice(result)
        for value in list(result['implementations'].values()) + so2_rows + cueq_rows:
            if value.get('status') == 'passed':
                require_exclusive(value.get('exclusive_gpu_proof'))
                devices |= gpu_ids(value['exclusive_gpu_proof'])
                if not value.get('feature_layout') or not value.get('memory_scope'):
                    raise ValueError('Native layout and isolated implementation memory evidence are required')
                if value['memory_scope'].get('other_implementations') != 'no live parameters, inputs or geometry':
                    raise ValueError('Other implementation tensors contaminated the memory measurement')
        if not result.get('equivalence', {}).get('passed'):
            raise ValueError('Operator case failed its numerical check')
        row['so2cuda_alternatives'] = [implementation(value) for value in so2_rows]
        row['cueq_alternatives'] = [implementation(value) for value in cueq_rows]
        general = next(value for value in so2_rows if value['candidate'] == TABLE_ENTRY)
        row['implementations']['so2cuda'] = implementation(general)
        row['so2cuda_entry'] = 'so2_cuda_ops.deeptb.true_dense_pairs(include_m0=True)'
        row['entries'] = {value['candidate']: implementation(value) for value in so2_rows}
        if 'activation' in result['implementations']:
            activation = result['implementations']['activation']
            if activation.get('status') == 'passed':
                require_exclusive(activation.get('exclusive_gpu_proof'))
                if result['equivalence'].get('implementations', {}).get('activation', {}).get('status') != 'passed':
                    raise ValueError('activation_forward entry lacks its numerical check')
            row['entries']['activation'] = implementation(activation)
        row['equivalence'] = metrics(result['equivalence'])
        checks.append({'case': result['equivalence'], 'so2cuda': [value.get('equivalence') for value in so2_rows],
                       'cueq': [value.get('equivalence') for value in cueq_rows]})
        cases.append(row)
    report = {'schema': 'so2cuda-public-operator-benchmarks-v2', 'hardware': 'NVIDIA H200',
              'precision': 'FP32', 'tf32': False, 'geometry_precomputed': True,
              'timing_contract': 'native-layout-v2', 'feature_layout_conversion_timed': False,
              'single_device_session': len(devices) == 1, 'source_commits': commits(source), 'cases': cases}
    return report, checks


def model_applicability(raw):
    if (raw.get('schema') != 'ob-model-eqv3-applicability-v1' or raw.get('status') != 'complete'
            or set(raw.get('models', {})) != {'dense', 'unitb'}):
        raise ValueError('Both model EquiformerV3 applicability receipts are required')
    result = {'schema': raw['schema'], 'models': {}}
    for name, model in raw['models'].items():
        applicable = model.get('applicable')
        if not isinstance(applicable, bool) or not model.get('layers'):
            raise ValueError('Model applicability lacks inspected layers')
        row = {'applicable': applicable, 'status': model['status'], 'layers': []}
        if not applicable:
            if model.get('status') != 'not_applicable' or not model.get('reason_zh'):
                raise ValueError('Model N/A lacks a concrete reason')
            row['reason'] = public_reason(model['reason_zh'])
        for layer in model['layers']:
            entry = {'applicable': bool(layer['applicable'])}
            for direction in ('input', 'output'):
                side = layer[direction]
                entry[direction] = {'channels_by_l': side['channels_by_l'], 'uniform': bool(side['uniform'])}
            for key in ('extra_m0_api_shape_possible', 'extra_m0_out_channels'):
                if key in layer:
                    entry[key] = layer[key]
            row['layers'].append(entry)
        result['models'][name] = row
    return result


def model_evidence(raw, applicability, operator_source):
    if raw.get('schema') != 'h200-model-session-v1' or raw.get('status') != 'completed':
        raise ValueError('Model session summary has not completed')
    if raw.get('payload_source', {}).get('commits') != operator_source.get('commits'):
        raise ValueError('Operator and model sessions measured different sources')
    expected = {case['id'] for case in raw['matrix']['cases'] if case['kind'] == 'model'}
    received = [case.get('id') for case in raw['cases']]
    if len(set(received)) != len(received) or set(received) != expected:
        raise ValueError('Model session does not contain every matrix case exactly once')
    cases, devices = {}, set()
    for wrapper in raw['cases']:
        report, configuration = wrapper.get('result'), wrapper['configuration']
        if wrapper.get('status') not in ('completed', 'oom') or report is None:
            raise ValueError('Model case is incomplete or failed')
        if report.get('schema') != 'h200-model-case-v1' or 'H200' not in report.get('gpu', {}).get('name', ''):
            raise ValueError('Public model timings must be measured on H200')
        if report.get('warmup_iterations') != 3 or report.get('measured_iterations') != 10:
            raise ValueError('Model columns must use the three/ten protocol')
        precision = report.get('precision', {})
        if precision.get('dtype') != 'float32' or precision.get('allow_tf32') is not False or precision.get('cudnn_allow_tf32') is not False:
            raise ValueError('Model precision evidence is missing')
        require_exclusive(report.get('exclusive_gpu_proof'))
        devices |= gpu_ids(report['exclusive_gpu_proof'])
        column = configuration['column']
        if column not in MODEL_COLUMNS:
            continue
        value = report['backends'][column]
        if value.get('status') == 'passed':
            if column == 'cueq' and not value.get('shared_wigner', {}).get('passed'):
                raise ValueError('cuEquivariance model timing lacks shared-Wigner execution evidence')
            if column == 'so2cuda' and not value.get('route_coverage'):
                raise ValueError('SO2CUDA model timing lacks route coverage evidence')
        key = (configuration['model'], configuration['edges'])
        entry = cases.setdefault(key, {'model': configuration['model'], 'edges': configuration['edges'],
                                       'shape': report['shape'], 'implementations': {}})
        if column in entry['implementations']:
            raise ValueError('Duplicate model timing')
        entry['implementations'][column] = implementation(value)
    for row in cases.values():
        if set(row['implementations']) != set(MODEL_COLUMNS):
            raise ValueError('Model case lacks a column')
        audit = applicability['models'][row['model']]
        if audit['applicable']:
            raise ValueError('An applicable EquiformerV3 model column is not measured')
        row['implementations']['eqv3'] = {'status': 'N/A', 'reason': audit['reason']}
    order = sorted(cases.values(), key=lambda row: (row['model'] != 'dense', row['edges']))
    return {'schema': 'so2cuda-public-model-benchmarks-v2', 'hardware': 'NVIDIA H200',
            'precision': 'FP32', 'tf32': False, 'warmup': 3, 'iterations': 10,
            'single_device_session': len(devices) == 1,
            'source_commits': commits(raw['payload_source']), 'cases': order,
            'eqv3_applicability': applicability}, devices


def model_equivalence(raw_so2cuda, raw_cueq, root):
    """SO2CUDA against our pure PyTorch, and the cuEquivariance route against it, whole model."""
    so2 = metrics(raw_so2cuda)
    if not so2['all_passed']:
        raise ValueError('SO2CUDA whole-model equivalence failed')
    for name, model in raw_so2cuda.get('models', {}).items():
        if not model.get('differences'):
            raise ValueError('SO2CUDA whole-model equivalence lacks comparisons: ' + name)
    result = {'so2cuda_vs_pure_pytorch': so2}
    if raw_cueq is not None:
        if raw_cueq.get('status') != 'passed':
            raise ValueError('cuEquivariance whole-model equivalence must pass')
        for name in CUEQ_MODEL_FILES:
            if raw_cueq.get('source_file_sha256', {}).get(name) != digest(root / 'examples' / name):
                raise ValueError('cuEquivariance model evidence used different adapter source: ' + name)
        result['cueq_vs_pure_pytorch'] = metrics(raw_cueq)
        if not result['cueq_vs_pure_pytorch']['all_passed']:
            raise ValueError('cuEquivariance whole-model equivalence failed')
    return result


def value_cell(row, accelerated=None):
    status = row['status']
    if status != 'passed':
        return 'OOM' if status == 'oom' else 'N/A'
    value = row['forward_backward']['median_ms']
    text = f'{value:.2f}'
    if accelerated is not None and accelerated['status'] == 'passed':
        text += f'（{value / accelerated["forward_backward"]["median_ms"]:.2f}×）'
    return text


def config_label(config, group):
    if group in ('A', 'B', 'D'):
        return f'ℓ={config["lmax"]}, C={config["channels"]}, m≤{config["mmax"]}'
    return 'V 形通道' if config['irreps_in'] == config['irreps_out'] else '输入／输出 irreps 不同'


OPERATOR_NAMES = ['eqv3', 'eqv3+compile', 'naive', 'cueq', 'so2cuda']


def operator_tables(report):
    lines = []
    for group, title in (('A', '形状'), ('B', '规模'), ('C', '一般 irreps'), ('D', '截断 m')):
        rows = [row for row in report['cases'] if group in row['grids']]
        if not rows:
            continue
        rows.sort(key=lambda row: (row['config'].get('lmax', 0), row['config'].get('channels', 0),
                                  row['config'].get('irreps_in', ''), row['config']['edges']))
        lines += [f'**{title}**', '', '| 配置 | 有向边 | ' + ' | '.join(
            LABELS[name] + (' ms' if name == 'so2cuda' else ' ms（时间比）') for name in OPERATOR_NAMES) + ' |',
            '|---|---:|' + '---:|' * len(OPERATOR_NAMES)]
        for row in rows:
            accelerated = row['implementations']['so2cuda']
            cells = [value_cell(row['implementations'].get(name, {'status': 'N/A'}),
                                accelerated if name != 'so2cuda' else None) for name in OPERATOR_NAMES]
            lines.append(f'| {config_label(row["config"], group)} | {row["config"]["edges"]:,} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['<details>', '<summary>前向计时、四分位与峰值显存</summary>', '',
              '| 配置 | 有向边 | 实现 | 前向 ms（Q1–Q3） | 前向＋反向 ms（Q1–Q3） | 前向＋反向峰值 GiB |',
              '|---|---:|---|---:|---:|---:|']
    for row in report['cases']:
        for name in OPERATOR_NAMES:
            impl = row['implementations'].get(name, {'status': 'N/A'})
            if impl['status'] != 'passed':
                continue
            fwd, both = impl['forward'], impl['forward_backward']
            peak = both.get('peak_allocated_bytes')
            memory = f'{peak / 2 ** 30:.2f}' if peak is not None else '—'
            lines.append(f'| {config_label(row["config"], row["grids"][0])} | {row["config"]["edges"]:,} | {LABELS[name]} | '
                f'{fwd["median_ms"]:.2f}（{fwd["q1_ms"]:.2f}–{fwd["q3_ms"]:.2f}） | '
                f'{both["median_ms"]:.2f}（{both["q1_ms"]:.2f}–{both["q3_ms"]:.2f}） | {memory} |')
    lines += ['', '全部实现与 SO2CUDA 各公开入口的计时、数值核对与报错见 [算子 JSON](docs/benchmarks/OP_SPEED_H200.json)。',
              '', '</details>']
    return '\n'.join(lines)


def entry_table(report):
    """Forward plus backward medians of every SO2CUDA public entry, for the operator documentation."""
    lines = ['| 配置 | 有向边 | ' + ' | '.join(label + ' ms' for _, label in ENTRY_LABELS) + ' |',
             '|---|---:|' + '---:|' * len(ENTRY_LABELS)]
    spread = []
    for row in report['cases']:
        base = row['entries'][TABLE_ENTRY]
        cells = []
        for key, _ in ENTRY_LABELS:
            value = row['entries'].get(key, {'status': 'N/A'})
            cells.append(value_cell(value))
            if value['status'] == 'passed' and base['status'] == 'passed' and key != TABLE_ENTRY:
                spread.append(value['forward_backward']['median_ms'] / base['forward_backward']['median_ms'])
        lines.append(f'| {config_label(row["config"], row["grids"][0])} | {row["config"]["edges"]:,} | '
                     + ' | '.join(cells) + ' |')
    if spread:
        lines += ['', f'其他入口相对 `true_dense_pairs` 的前向＋反向时间比为 {min(spread):.3f}–{max(spread):.3f}。']
    return '\n'.join(lines)


def model_table(report):
    names = ['eqv3', 'naive', 'cueq', 'so2cuda']
    lines = ['| 模型 | 有向边 | ' + ' | '.join(
                LABELS[name] + (' ms' if name in ('so2cuda', 'eqv3') else ' ms（时间比）') for name in names) + ' |',
             '|---|---:|' + '---:|' * len(names)]
    for row in report['cases']:
        impls = row['implementations']
        accelerated = impls['so2cuda']
        cells = [value_cell(impls[name], accelerated if name in ('naive', 'cueq') else None) for name in names]
        label = 'UniTB-dense' if row['model'] == 'dense' else 'UniTB'
        lines.append(f'| {label} | {row["edges"]:,} | ' + ' | '.join(cells) + ' |')
    lines += ['', '<details>', '<summary>前向计时、四分位与峰值显存</summary>', '',
              '| 模型 | 有向边 | 实现 | 前向 ms（Q1–Q3） | 前向＋反向 ms（Q1–Q3） | 峰值 GiB |', '|---|---:|---|---:|---:|---:|']
    for row in report['cases']:
        label = 'UniTB-dense' if row['model'] == 'dense' else 'UniTB'
        for name in ('naive', 'cueq', 'so2cuda'):
            impl = row['implementations'][name]
            if impl['status'] != 'passed':
                continue
            fwd, both = impl['forward'], impl['forward_backward']
            peak = both.get('peak_allocated_bytes')
            memory = f'{peak / 2 ** 30:.2f}' if peak is not None else '—'
            lines.append(f'| {label} | {row["edges"]:,} | {LABELS[name]} | '
                         f'{fwd["median_ms"]:.2f}（{fwd["q1_ms"]:.2f}–{fwd["q3_ms"]:.2f}） | '
                         f'{both["median_ms"]:.2f}（{both["q1_ms"]:.2f}–{both["q3_ms"]:.2f}） | {memory} |')
    lines += ['', '</details>']
    return '\n'.join(lines)


def factual_summary(cases, names):
    words = {'eqv3': '与 EquiformerV3 原版相比', 'eqv3+compile': '与 EquiformerV3 + compile 相比',
             'naive': '与我们的纯 PyTorch 实现相比', 'cueq': '与 cuEquivariance 相比'}
    lines = []
    for name in names:
        pairs = [(row['implementations'][name]['forward_backward']['median_ms'],
                  row['implementations']['so2cuda']['forward_backward']['median_ms'])
                 for row in cases if row['implementations'].get(name, {}).get('status') == 'passed'
                 and row['implementations']['so2cuda']['status'] == 'passed']
        if not pairs:
            continue
        ratios = [a / b for a, b in pairs]
        faster = sum(b < a for a, b in pairs)
        slower = sum(b > a for a, b in pairs)
        span = f'{min(ratios):.2f}–{max(ratios):.2f}'
        if faster == len(pairs):
            verdict = f'{len(pairs)} 个配置中 SO2CUDA 都更快'
        elif slower == len(pairs):
            verdict = f'{len(pairs)} 个配置中 SO2CUDA 都更慢'
        else:
            verdict = f'{len(pairs)} 个配置中 SO2CUDA 更快 {faster} 个、更慢 {slower} 个'
        lines.append(f'- {words[name]}：{verdict}，对方用时为 SO2CUDA 的 {span} 倍。')
    return '\n'.join(lines)


def model_context(readme):
    if CONTEXT_BEGIN not in readme or CONTEXT_END not in readme:
        raise ValueError('Existing model benchmark context is missing')
    text = readme.split(CONTEXT_BEGIN, 1)[1].split(CONTEXT_END, 1)[0].strip()
    text = text.replace(CLARIFICATION, '').strip()
    return text + '\n\n' + CLARIFICATION


def render(operator, model, equiv_op, equiv_model, context):
    versions = sorted({value['version'] for row in operator['cases']
                       for name, value in row['implementations'].items() if name == 'cueq' and value.get('version')})
    if len(versions) != 1:
        raise ValueError('Exactly one measured cuEquivariance version is required')
    choices = {(row['implementations']['cueq'].get('descriptor'), row['implementations']['cueq'].get('method'),
                row['implementations']['cueq'].get('rotation')) for row in operator['cases']
               if row['implementations']['cueq']['status'] == 'passed'}
    if len(choices) != 1:
        raise ValueError('The cuEquivariance column must use one combination')
    descriptor, method, rotation = next(iter(choices))
    warmup = min(row['warmup'] for row in operator['cases'])
    iterations = min(row['iterations'] for row in operator['cases'])
    if not equiv_op['all_passed'] or not all(value['all_passed'] for value in equiv_model.values()):
        raise ValueError('Numerical equivalence evidence failed')
    same_device = operator['single_device_session'] and model['single_device_session']
    session = ('全部配置与实现在同一块 NVIDIA H200 上的同一次会话中测量' if same_device
               else 'NVIDIA H200')
    uniform = [row for row in operator['cases'] if 'C' not in row['grids']]
    lines = [BEGIN, '## 性能测试', '', '### 1. SO(2) 张量积算子', '',
        '边上的 SO(2) 卷积先将输入旋转到边的主轴，做按 |m| 分组的共享线性变换，再旋转回原坐标：', '',
        r'$$y_e=D(R_e)^\top\,\mathcal{L}_W\!\left(D(R_e)x_e\right).$$', '',
        '`m=0` 使用实线性变换，`m>0` 使用 2×2 复结构；边之间共享权重，算子测试不含径向调制。', '',
        'EquiformerV3 原版本身就是显式旋转：把系数排列并入稠密 Wigner 矩阵后用 `bmm` 旋转，再做逐 m 线性和逆旋转。'
        'SO2CUDA 与它数学相同，差别在 indexed sandwich 的实现：CUDA kernel 以（边，通道）为单位，把输入的各 ℓ 分量旋入每个 m '
        '一块的缓冲；每块做一次 GEMM，m>0 块的 2×2 复结构写成一个实数块权重；再从各块旋回并写出特征。'
        '反向由同样的 kernel 与 GEMM 完成。我们的纯 PyTorch 实现按 DeePTB 上游写法逐 l 旋转、按 m 选取特征后做线性层。', '',
        '- SO2CUDA：通用入口 `true_dense_pairs(..., include_m0=True)`，一次调用算完整层（含 m=0）。'
        '其他公开入口（`dense_pairs`、单专家的 `activation_forward`）执行同一套 kernel，'
        '各入口逐配置的计时见 [docs/operator-benchmark.md](docs/operator-benchmark.md#各入口计时)。',
        '- 我们自己的纯 PyTorch 实现：按 DeePTB 上游 `SO2_Linear` 写法实现的算子，见 '
        '[operator_baselines.py](examples/operator_baselines.py)。',
        '- EquiformerV3 原版：[固定提交](https://github.com/atomicarchitects/equiformer_v3/tree/'
        'a7300c58df683dc99cb48027d5bfd4c887486c48) 的 `SO3Rotation` ＋ `SO2Linear`（eager，源码不改），`fc_m0` 的 bias 置零；'
        f'另列同一实现经 `torch.compile` 后的时间（各 ℓ 通道数一致的 {len(uniform)} 个配置）。',
        f'- cuEquivariance {versions[0]}：`SO3` irreps 的 `{descriptor}` 描述符由 `SegmentedPolynomial` 执行，method 为 '
        f'`{method}`，旋转方式为 `{rotation}`；这一组合在全部描述符／method／旋转组合中前向＋反向最快且数值正确。', '',
        f'{session}，计时前后核验独占，严格 FP32、关闭 TF32；几何量按各实现的格式预先计算。'
        '各实现使用各自原生特征布局，布局转换在计时区外；峰值显存只保留当前实现的输入、参数和几何量。'
        f'每路预热 {warmup} 次、计时 {iterations} 次。下表为前向＋输入和全部权重反向的 ms 中位数；'
        '括号内为该实现时间 ÷ SO2CUDA 时间：大于 1 表示 SO2CUDA 更快，小于 1 表示该实现更快。'
        'SO2CUDA 列均为 `true_dense_pairs(include_m0=True)`。', '',
        factual_summary(operator['cases'], ('eqv3', 'eqv3+compile', 'naive', 'cueq')), '', operator_tables(operator), '',
        '每个配置计时前，各实现（含 SO2CUDA 的各公开入口）先在 128 条边上与 FP64 参考比较输出、输入梯度和映射回共同参数的'
        '权重梯度，并做整体旋转等变检查，均通过 FP32 的绝对误差／相对 L2 联合判据；近零参考量使用绝对误差判据。'
        '逐项误差和判据见 [算子数值证据](docs/benchmarks/EQUIV_OP_H200.json)。', '',
        '各实现的参数映射与计时约定见 [docs/operator-benchmark.md](docs/operator-benchmark.md)。', '',
        '一般 irreps 表中 EquiformerV3 的两列为 N/A：原实现要求各 ℓ 的通道数一致；这些行的时间比相对我们的纯 PyTorch 和 '
        'cuEquivariance 报告。', '',
        '复现算子测试（在本仓库根目录）：', '', '```bash',
        f'pip install cuequivariance-torch=={versions[0]}',
        'git clone https://github.com/atomicarchitects/equiformer_v3.git && git -C equiformer_v3 checkout a7300c58df683dc99cb48027d5bfd4c887486c48',
        'python examples/so2_operator_speed_test.py --impl naive,so2cuda,eqv3,cueq --include-compile '
        f'--cueq-choice {descriptor},{method},{rotation} --lmax 6 --channels 128 --edges 50000 '
        '--warmup 5 --iterations 20 --eqv3-root equiformer_v3 --json operator.json',
        '```', '', '### 2. UniTB 模型', '', CONTEXT_BEGIN, context, CONTEXT_END, '',
        ('与上节算子测试在同一块 NVIDIA H200 上的同一次会话中测量' if same_device else 'NVIDIA H200')
        + '，严格 FP32，每次计时迭代前后核验独占；每路先做一次不计时的前向＋反向，再预热 3 次、计时 10 次。'
        '下表为前向＋反向 ms 中位数，括号内为该实现时间 ÷ SO2CUDA 时间。'
        f'cuEquivariance 列在全部六个 SO(2) 层统一使用 `{descriptor}` 描述符与 `{method}` method，每次前向共享一次 Wigner 构建。', '',
        model_table(model), '',
        *[('UniTB-dense' if name == 'dense' else 'UniTB') + ' 的 EquiformerV3 为 N/A：' + row['reason']
          for name, row in model['eqv3_applicability']['models'].items() if not row['applicable']], '',
        'SO2CUDA 与 cuEquivariance 两条路线的训练输出、推理输出和全部参数梯度均与我们的纯 PyTorch 实现比较，'
        '全部有限并通过绝对误差／相对 L2 联合判据；近零梯度使用绝对误差判据。', '',
        'cuEquivariance 路线以标准子模块替换 SO(2) 算子，路由器、电荷头和其余模型层共享；旋转用分块 bmm，'
        '再按描述符的原生布局执行线性层。N/A 表示该路线未提供忠实的对应实现；合成模型结果不代表物理精度或真实训练吞吐。', '',
        '完整计时、四分位与误差见 [模型 JSON](docs/benchmarks/MODEL_SPEED_H200.json) 和 '
        '[模型数值证据](docs/benchmarks/EQUIV_MODEL.json)。', END]
    return '\n'.join(lines) + '\n'


def replace_section(readme, section):
    if BEGIN not in readme or END not in readme:
        raise ValueError('README benchmark markers are missing')
    start = readme.index(BEGIN)
    finish = readme.index(END, start) + len(END)
    return readme[:start] + section.rstrip() + readme[finish:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-json', type=Path, required=True, help='Operator session summary')
    parser.add_argument('--model-json', type=Path, required=True, help='Whole-model session summary')
    parser.add_argument('--model-eqv3-json', type=Path, required=True, help='EquiformerV3 model applicability audit')
    parser.add_argument('--equiv-model-json', type=Path, required=True,
                        help='Whole-model SO2CUDA against pure PyTorch comparison (deeptb_speed_test.py --backend both)')
    parser.add_argument('--equiv-model-cueq-json', type=Path,
                        help='Whole-model cuEquivariance (uniform compact descriptor) against pure PyTorch comparison')
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--readme', type=Path, default=root / 'README.md')
    parser.add_argument('--operator-doc', type=Path, default=root / 'docs/operator-benchmark.md')
    parser.add_argument('--docs-dir', type=Path, default=root / 'docs/benchmarks')
    parser.add_argument('--write-readme', action='store_true', help='Apply the generated section to README')
    args = parser.parse_args()
    raw_operator, raw_model = load(args.operator_json), load(args.model_json)
    operator, checks = operator_evidence(raw_operator)
    applicability = model_applicability(load(args.model_eqv3_json))
    model, _ = model_evidence(raw_model, applicability, raw_operator['payload_source'])
    equiv_op = metrics(checks)
    equiv_model = model_equivalence(load(args.equiv_model_json),
                                    load(args.equiv_model_cueq_json) if args.equiv_model_cueq_json else None, root)
    inputs = {'operator': digest(args.operator_json), 'model': digest(args.model_json),
              'model_eqv3': digest(args.model_eqv3_json), 'equiv_model': digest(args.equiv_model_json)}
    if args.equiv_model_cueq_json:
        inputs['equiv_model_cueq'] = digest(args.equiv_model_cueq_json)
    readme = args.readme.read_text(encoding='utf-8')
    section = render(operator, model, equiv_op, equiv_model, model_context(readme))
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (('OP_SPEED_H200', operator), ('MODEL_SPEED_H200', model),
                        ('EQUIV_OP_H200', equiv_op), ('EQUIV_MODEL', equiv_model),
                        ('MODEL_EQV3_APPLICABILITY', applicability)):
        (args.docs_dir / (name + '.json')).write_text(
            json.dumps({**value, 'input_sha256': inputs}, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8')
    (args.docs_dir / 'README_SECTION.md').write_text(section, encoding='utf-8')
    entries = entry_table(operator)
    if args.write_readme:
        args.readme.write_text(replace_section(readme, section), encoding='utf-8')
        doc = args.operator_doc.read_text(encoding='utf-8')
        if ENTRIES_BEGIN not in doc or ENTRIES_END not in doc:
            raise ValueError('Operator documentation lacks the entry timing markers')
        start = doc.index(ENTRIES_BEGIN) + len(ENTRIES_BEGIN)
        args.operator_doc.write_text(doc[:start] + '\n' + entries + '\n' + doc[doc.index(ENTRIES_END, start):],
                                     encoding='utf-8')
    print(json.dumps({'public_json_files': 5, 'readme_updated': args.write_readme}, ensure_ascii=False))


if __name__ == '__main__':
    main()
