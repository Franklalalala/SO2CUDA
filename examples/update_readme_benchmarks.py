#!/usr/bin/env python3
"""Generate Chinese benchmark tables and public, allowlisted JSON evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import itertools


BEGIN = '<!-- SO2CUDA_BENCHMARKS_BEGIN -->'
END = '<!-- SO2CUDA_BENCHMARKS_END -->'
CONTEXT_BEGIN = '<!-- SO2CUDA_MODEL_CONTEXT_BEGIN -->'
CONTEXT_END = '<!-- SO2CUDA_MODEL_CONTEXT_END -->'
LABELS = {'so2cuda': 'SO2CUDA', 'naive': '我们的纯 PyTorch', 'eqv3': 'EquiformerV3 原版',
          'eqv3+compile': 'EquiformerV3 + compile', 'cueq': 'cuEquivariance',
          'explicit_gemm': '显式旋转＋SO2CUDA GEMM'}
METHODS = {'naive', 'uniform_1d', 'fused_tp', 'indexed_linear'}
DESCRIPTORS = {'escn_tp', 'escn_tp_compact'}
SO2CUDA_CANDIDATES = {'dense_pairs', 'dense_pairs_grouped', 'true_dense_pairs'}
CLARIFICATION = ('“我们的纯 PyTorch”是我们按 DeePTB 上游 `SO2_Linear` 与 UMA MoLE 写法自己实现的'
    '（[examples/naive_baseline.py](examples/naive_baseline.py)）。朴素 PyTorch SO(2) 基线采用 '
    'EquiformerV3 原版 `SO3Rotation` ＋ `SO2Linear`（eager）；模型层的通道布局不适用时标 N/A。')
ABLATION_SHAPES = {(2, 128, 50000), (4, 128, 50000), (6, 128, 50000),
                   (6, 32, 20000), (6, 32, 50000), (6, 32, 130000)}


def public_reason(value):
    """Preserve diagnostics while removing private path and host identities."""
    if not isinstance(value, str):
        raise ValueError('Diagnostic reason must be text')
    value = re.sub(r'(?:/[A-Za-z0-9_.+~-]+){2,}', '<path>', value)
    value = re.sub(r'\b(?:[A-Za-z]:\\)[^\s\"\']+', '<path>', value)
    value = re.sub(r'\b[^\s@]+@[^\s@]+\b', '<identity>', value)
    value = re.sub(r'GPU-[A-Za-z0-9-]+', '<gpu>', value)
    return value


def ablation_shape(config):
    return (config.get('lmax'), config.get('channels'), config.get('edges')) in ABLATION_SHAPES \
        and config.get('mmax') == config.get('lmax')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected finite numeric evidence, got {value!r}')
    return value


def commits(value):
    found = {}
    aliases = {'so2cuda': 'SO2CUDA', 'deeptb': 'DeePTB', 'equiformerv3': 'EquiformerV3',
               'eqv3': 'EquiformerV3'}
    for row in (value.get('payload_source', {}).get('commits', {}), value.get('payload_commits', {}),
                value.get('provenance', {}).get('commits', {})):
        for key, sha in row.items():
            name = aliases.get(key.lower().replace('_', ''))
            if name and isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{40}', sha):
                found[name] = sha
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
            # Rejected tuning candidates are diagnostic evidence, not failures
            # of the numerically qualified implementation used in the table.
            if node.get('required') is False and node.get('status') != 'passed':
                return
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
                if key in ('method_selection', 'cueq_alternatives', 'rejected_candidate_evidence'):
                    continue
                visit(child, key if key in quantities else quantity)
        elif isinstance(node, list):
            for child in node:
                visit(child, quantity)

    visit(value)
    if not rows:
        raise ValueError('Equivalence evidence contains no numerical metrics')
    return {'metric_count': len(rows), 'all_passed': all(row.get('passed', True) and row.get('finite', True) for row in rows),
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
                       'so2_cuda_ops.deeptb.true_dense_pairs',
                       'torch.bmm + so2_cuda_ops.grouped_gemm_multi + torch.bmm')))
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
    if row.get('reused_rejection'):
        reused = row['reused_rejection']
        digest = reused.get('evidence_sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Reused rejection lacks an evidence hash')
        result['reused_rejection'] = {'evidence_sha256': digest,
            'reason': public_reason(reused['reason'])}
        provenance = reused.get('source_provenance') or {}
        revision = provenance.get('so2cuda_sha')
        if isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40}', revision):
            result['reused_rejection']['so2cuda_sha'] = revision
    if row.get('shared_wigner'):
        result['shared_wigner'] = {key: number(row['shared_wigner'][key]) for key in
            ('expected_model_forwards', 'geometry_builds', 'cache_hits', 'so2_layers')
            if key in row['shared_wigner']}
        result['shared_wigner']['passed'] = row['shared_wigner'].get('passed') is True
    if 'reason' in row:
        result['reason'] = public_reason(row['reason'])
    elif row.get('error'):
        result['reason'] = public_reason(str(row['error']))
    if row.get('error_type'):
        result['error_type'] = public_reason(row['error_type'])
    if row.get('environment'):
        result['environment'] = {key: public_reason(value) for key, value in row['environment'].items()
            if key in ('SO2_CUDA_FORWARD_MODE', 'DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE') and isinstance(value, str)}
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
            entry = {'descriptor': layer['descriptor'], 'method': layer['method'],
                     'rotation': layer['rotation']}
            for key in ('rotation_detail', 'model_feature_layout', 'descriptor_feature_layout', 'wigner_layout'):
                if key in layer:
                    entry[key] = public_reason(layer[key])
            if entry['method'] not in METHODS or entry['descriptor'] not in DESCRIPTORS or entry['rotation'] != 'pytorch':
                raise ValueError('Unknown model descriptor method')
            selection = layer.get('method_selection') or {}
            entry['candidates'] = []
            for candidate in selection.get('candidates', []):
                item = {'descriptor': candidate['descriptor'], 'method': candidate['requested_method'],
                        'status': candidate['status']}
                if (item['method'] not in METHODS or item['descriptor'] not in DESCRIPTORS
                        or item['status'] not in ('passed', 'unsupported', 'unsupported_or_incorrect', 'unavailable', 'oom', 'failed_equivalence')):
                    raise ValueError('Unknown model descriptor candidate')
                for key in ('forward_backward_median_ms', 'forward_backward_q1_ms',
                            'forward_backward_q3_ms', 'forward_median_ms', 'initialization_and_check_seconds'):
                    if key in candidate:
                        item[key] = number(candidate[key])
                if 'correctness_max_abs' in candidate:
                    item['correctness_max_abs'] = [number(value) for value in candidate['correctness_max_abs']]
                if candidate.get('equivalence'):
                    item['equivalence'] = metrics(candidate['equivalence'])
                if candidate.get('error') or candidate.get('reason'):
                    item['reason'] = public_reason(candidate.get('error') or candidate['reason'])
                entry['candidates'].append(item)
            layers.append(entry)
        result['cueq_layers'] = layers
    return result


def selection_evidence(report, scope):
    """Retain rejected candidate measurements apart from accepted summaries."""
    selections = []
    if scope == 'model':
        for name, model in report.get('models', {}).items():
            if name not in ('dense', 'unitb'):
                raise ValueError('Unknown public model name')
            layers = implementation({'status': 'N/A', 'cueq_execution': model.get('cueq_execution', {})})
            selections.append({'model': name, 'layers': layers.get('cueq_layers', [])})
    else:
        candidate_rows = list(report.get('optimized', {}).values())
        candidate_rows += list(report.get('rejected_candidate_evidence', {}).get('candidates', {}).values())
        for candidate in candidate_rows:
            entry = {key: candidate[key] for key in ('descriptor', 'method', 'rotation', 'status')}
            if entry['descriptor'] not in DESCRIPTORS or entry['method'] not in METHODS or entry['rotation'] not in ('pytorch', 'cueq'):
                raise ValueError('Unknown optimized operator candidate')
            if candidate.get('result'):
                entry['equivalence'] = metrics(candidate['result'])
            if candidate.get('error') or candidate.get('reason'):
                entry['reason'] = public_reason(candidate.get('error') or candidate['reason'])
            if candidate.get('error_type'):
                entry['error_type'] = public_reason(candidate['error_type'])
            selections.append(entry)
    return selections


def require_exclusive(proof):
    if not isinstance(proof, list) or len(proof) < 2 or any(row.get('exclusive') is not True for row in proof):
        raise ValueError('Missing before/after exclusive GPU evidence')


def frozen_source(operator, model, inputs):
    first, second = operator.get('payload_source', {}), model.get('payload_source', {})
    for key in ('commits', 'cueq_version', 'matrix_sha256', 'runners', 'runtime_files',
                'timing_contract', 'cueq_rejection_files',
                'model_equivalence_sha256', 'model_eqv3_applicability_sha256',
                'cueq_target_files', 'existing_model_columns',
                'model_compact_naive_equivalence_sha256', 'model_compact_fused_equivalence_sha256',
                'operator_candidate_equivalence_sha256',
                'followup_manifest_sha256', 'followup_prior_operator_sha256', 'followup_prior_model_sha256'):
        if first.get(key) != second.get(key):
            raise ValueError('Operator and model evidence used different frozen sources')
    if set(commits(operator)) != {'SO2CUDA', 'DeePTB', 'EquiformerV3'}:
        raise ValueError('Frozen source commit provenance is incomplete')
    if first.get('model_equivalence_sha256') != inputs['equiv_model']:
        raise ValueError('Model equivalence receipt differs from the frozen payload')
    if first.get('model_eqv3_applicability_sha256') != inputs['model_eqv3']:
        raise ValueError('Model EquiformerV3 applicability differs from the frozen payload')
    old = first.get('existing_model_columns', {})
    if old.get('retimed') is not False or old.get('sha256') != inputs['old_model']:
        raise ValueError('Existing model timing source differs from the frozen payload')
    if first.get('model_compact_fused_equivalence_sha256'):
        if first['model_compact_fused_equivalence_sha256'] != inputs.get('equiv_model_fused'):
            raise ValueError('Uniform compact/fused model equivalence differs from frozen payload')
        if first.get('model_compact_naive_equivalence_sha256') != inputs['equiv_model']:
            raise ValueError('Uniform compact/naive model equivalence differs from frozen payload')
    if first.get('operator_candidate_equivalence_sha256') != inputs.get('equiv_op_candidates'):
        raise ValueError('SO2CUDA candidate numerical proof differs from frozen payload')
    if operator.get('measurement_sources'):
        if operator['measurement_sources'] != model.get('measurement_sources'):
            raise ValueError('Mixed timing sources disagree between operator and model evidence')
        if operator.get('followup_provenance') != model.get('followup_provenance'):
            raise ValueError('Mixed timing provenance differs between summaries')
        proof = operator['followup_provenance']
        for source_key, proof_key in (('followup_manifest_sha256', 'manifest_sha256'),
                ('followup_prior_operator_sha256', 'original_operator_sha256'),
                ('followup_prior_model_sha256', 'original_model_sha256')):
            if first.get(source_key) != proof.get(proof_key):
                raise ValueError('Mixed source does not pin the original inputs')


def column_sources(wrapper, summary):
    """Publish only hashes and commits for measurements retained across revisions."""
    result = {}
    for name, row in wrapper.get('implementation_sources', {}).items():
        source = summary.get('measurement_sources', {}).get(row.get('source'))
        if name not in LABELS or source is None or row.get('commits') != source.get('commits'):
            raise ValueError('Column measurement source is not bound to its frozen revision')
        if not isinstance(row.get('retimed'), bool):
            raise ValueError('Column timing provenance lacks a retiming declaration')
        if not re.fullmatch(r'[0-9a-f]{64}', row.get('receipt_sha256') or ''):
            raise ValueError('Column timing provenance lacks its raw receipt hash')
        result[name] = {key: row[key] for key in ('source', 'retimed', 'receipt_sha256', 'commits')}
    return result


def so2cuda_choices(source):
    rows = source.get('so2cuda_alternatives', [])
    if ({row.get('candidate') for row in rows} != SO2CUDA_CANDIDATES or len(rows) != 3):
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
    default = next(row for row in rows if row['candidate'] == 'dense_pairs')
    if source.get('ablation_so2cuda_default') != default:
        raise ValueError('Ablation endpoint must preserve the recorded default dense_pairs candidate')
    return rows


def cueq_choices(source):
    rows = source.get('cueq_alternatives', [])
    expected = set(itertools.product(DESCRIPTORS, METHODS, ('pytorch', 'cueq')))
    actual = {(row.get('descriptor'), row.get('method'), row.get('rotation')) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError('Each cuEquivariance descriptor/method/rotation candidate must be recorded once')
    choice = implementation(source['implementations']['cueq'])
    if choice['status'] == 'passed':
        for key in ('descriptor', 'method', 'rotation', 'version'):
            if key not in choice:
                raise ValueError('Selected cuEquivariance metadata is missing ' + key)
        valid = [row for row in rows if row.get('status') == 'passed']
        if not valid:
            raise ValueError('Selected cuEquivariance implementation has no passed candidates')
        winner = min(valid, key=lambda row: measurement(row, 'forward_backward')['median_ms'])
        if any(choice[key] != winner.get(key, winner.get('metadata', {}).get(key))
               for key in ('descriptor', 'method', 'rotation')):
            raise ValueError('Selected cuEquivariance implementation is not the fastest passed candidate')
        if measurement(winner, 'forward_backward') != choice['forward_backward']:
            raise ValueError('Selected cuEquivariance timing must reuse the recorded winner')
    return rows


def operator_evidence(raw):
    if raw.get('status') != 'completed':
        raise ValueError('Operator summary has not completed')
    if (raw.get('payload_source', {}).get('timing_contract') != 'native-layout-v2'
            or raw.get('matrix', {}).get('timing_contract') != 'native-layout-v2'):
        raise ValueError('Public timings require the native-layout-v2 measurement contract')
    expected = {case['id'] for case in raw.get('matrix', {}).get('cases', []) if case['kind'] == 'operator'}
    received = {case.get('id') for case in raw['cases']}
    if len(received) != len(raw['cases']):
        raise ValueError('Duplicate operator configuration receipts')
    if expected and expected != received:
        raise ValueError('Operator summary does not contain every frozen matrix case')
    cases = []
    for wrapper in raw['cases']:
        source = wrapper.get('result', wrapper)
        if source is None or source.get('status') not in ('completed', 'passed'):
            raise ValueError('Operator case is incomplete or failed')
        if source.get('timing_contract') != 'native-layout-v2':
            raise ValueError('Operator case used an old timing boundary')
        gpu = source.get('gpu', '')
        if 'H200' not in (gpu.get('name', '') if isinstance(gpu, dict) else gpu):
            raise ValueError('Public operator timings must be measured on H200')
        configuration = wrapper.get('configuration', source.get('config', {}))
        config = {key: configuration[key] for key in ('lmax', 'mmax', 'channels', 'edges') if key in configuration}
        for key in ('irreps_in', 'irreps_out'):
            value = source.get('config', {}).get(key, configuration.get(key))
            if value is not None and not re.fullmatch(r'[0-9xoe+ ]+', value):
                raise ValueError('Invalid public irreps representation')
            if value is not None:
                config[key] = value
        config.setdefault('mmax', source.get('config', {}).get('mmax'))
        config.setdefault('edges', source.get('config', {}).get('edges'))
        groups = configuration.get('grids', [])
        if not groups or any(group not in ('A', 'B', 'C', 'D') for group in groups):
            raise ValueError('Operator grid memberships are missing')
        row = {'config': config, 'grids': groups,
               'warmup': number(source['warmup']), 'iterations': number(source['iterations']),
               'implementations': {name: implementation(value) for name, value in source['implementations'].items()
                                   if name in LABELS}}
        if row['warmup'] < 5 or row['iterations'] < 20:
            raise ValueError('Require at least five warmups and twenty operator timing samples')
        if 'so2cuda' not in row['implementations']:
            raise ValueError('SO2CUDA comparison is missing')
        if source.get('precision') != 'strict FP32; TF32 disabled':
            raise ValueError('Operator precision evidence is missing')
        alternatives = cueq_choices(source)
        so2_alternatives = so2cuda_choices(source)
        for value in list(source['implementations'].values()) + alternatives + so2_alternatives:
            if value.get('status') == 'passed':
                require_exclusive(value.get('exclusive_gpu_proof'))
                if not value.get('feature_layout') or not value.get('memory_scope'):
                    raise ValueError('Native layout and isolated implementation memory evidence are required')
                if value['memory_scope'].get('other_implementations') != 'no live parameters, inputs or geometry':
                    raise ValueError('Other implementation tensors contaminated the memory measurement')
        row['cueq_alternatives'] = [implementation(value) for value in alternatives]
        row['so2cuda_alternatives'] = [implementation(value) for value in so2_alternatives]
        row['ablation_so2cuda_default'] = implementation(source['ablation_so2cuda_default'])
        row['implementation_sources'] = column_sources(wrapper, raw)
        if source.get('so2cuda_candidate_equivalence'):
            row['so2cuda_candidate_equivalence'] = metrics(source['so2cuda_candidate_equivalence'])
        if source.get('equivalence'):
            row['equivalence'] = metrics(source['equivalence'])
        if ablation_shape(config):
            explicit = row['implementations'].get('explicit_gemm')
            if not explicit:
                raise ValueError('Required explicit-GEMM ablation is missing')
            if explicit['status'] == 'passed':
                if source.get('equivalence', {}).get('implementations', {}).get('explicit_gemm', {}).get('status') != 'passed':
                    raise ValueError('Explicit-GEMM ablation was omitted from the numeric check')
                if not row.get('equivalence', {}).get('all_passed'):
                    raise ValueError('Explicit-GEMM ablation lacks passed equivalence')
                for name in ('eqv3', 'explicit_gemm', 'so2cuda'):
                    value = (row['ablation_so2cuda_default'] if name == 'so2cuda'
                             else row['implementations'].get(name))
                    if value and value['status'] == 'passed':
                        for scope in ('forward', 'forward_backward'):
                            if any(key not in value[scope] for key in ('peak_allocated_bytes', 'peak_reserved_bytes')):
                                raise ValueError('Ablation requires peak allocated and reserved memory in both scopes')
        cases.append(row)
    if {tuple(row['config'].get(key) for key in ('lmax', 'channels', 'edges'))
            for row in cases if ablation_shape(row['config'])} != ABLATION_SHAPES:
        raise ValueError('Required six unique ablation configurations are incomplete')
    return {'schema': 'so2cuda-public-operator-benchmarks-v1', 'hardware': 'NVIDIA H200',
            'precision': 'FP32', 'tf32': False, 'geometry_precomputed': True,
            'timing_contract': 'native-layout-v2', 'feature_layout_conversion_timed': False,
            'source_commits': commits(raw), 'cases': cases}


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
                source = layer[direction]
                entry[direction] = {'channels_by_l': source['channels_by_l'], 'uniform': bool(source['uniform'])}
            for key in ('extra_m0_api_shape_possible', 'extra_m0_out_channels'):
                if key in layer:
                    entry[key] = layer[key]
            row['layers'].append(entry)
        result['models'][name] = row
    return result


def model_evidence(old, new, applicability):
    if old.get('status') != 'completed' or new.get('status') != 'completed':
        raise ValueError('Model summary has not completed')
    expected = {case['id'] for case in new.get('matrix', {}).get('cases', []) if case['kind'] == 'model'}
    received = {case.get('id') for case in new['cases']}
    if len(received) != len(new['cases']):
        raise ValueError('Duplicate model configuration receipts')
    if expected and expected != received:
        raise ValueError('Model summary does not contain every frozen matrix case')
    cases = {}
    for source in old['cases']:
        key = (source['model'], source['requested_edges'])
        cases[key] = {'model': source['model'], 'edges': source['requested_edges'],
                      'implementations': {}}
        for old_name, name in (('reference', 'naive'), ('cuda', 'so2cuda')):
            cases[key]['implementations'][name] = implementation(source['backends'][old_name])
    for wrapper in new['cases']:
        source = wrapper.get('result', wrapper)
        if source is None or source.get('status') not in ('passed', 'oom'):
            raise ValueError('New model case is incomplete or failed')
        if 'H200' not in source.get('gpu', {}).get('name', ''):
            raise ValueError('Public model timings must be measured on H200')
        if source.get('warmup_iterations') != 3 or source.get('measured_iterations') != 10:
            raise ValueError('New model columns must use NB three/ten protocol')
        require_exclusive(source.get('exclusive_gpu_proof'))
        precision = source.get('precision', {})
        if precision.get('dtype') != 'float32' or precision.get('allow_tf32') is not False or precision.get('cudnn_allow_tf32') is not False:
            raise ValueError('Model precision evidence is missing')
        key = (source['model'], source['requested_edges'])
        if key not in cases:
            raise ValueError('New model column is outside the existing model matrix')
        for name, value in source['backends'].items():
            if name not in ('cueq', 'eqv3'):
                raise ValueError('Old model columns must not be retimed')
            if name in cases[key]['implementations']:
                raise ValueError('Duplicate new model timing')
            if name == 'cueq' and value.get('status') == 'passed' and not value.get('shared_wigner', {}).get('passed'):
                raise ValueError('New cueq model timing lacks shared-Wigner execution evidence')
            cases[key]['implementations'][name] = implementation(value)
        cases[key]['implementation_sources'] = column_sources(wrapper, new)
        candidates = source.get('cueq_alternatives', [])
        if (len(candidates) != 2 or {row.get('method') for row in candidates} != {'naive', 'fused_tp'}
                or any(row.get('descriptor') != 'escn_tp_compact' for row in candidates)):
            raise ValueError('Model cueq requires both uniform compact whole-model candidates')
        candidates_public = []
        for candidate in candidates:
            item = {key: candidate[key] for key in ('descriptor', 'method', 'rotation', 'status')}
            if item['status'] == 'passed':
                require_exclusive(candidate.get('exclusive_gpu_proof'))
                value = candidate.get('measurement') or {}
                if value.get('shared_wigner', {}).get('passed') is not True:
                    raise ValueError('Model cueq candidate lacks shared-Wigner proof')
                item['measurement'] = implementation(value)
                layer_rows = item['measurement'].get('cueq_layers', [])
                if len(layer_rows) != 6 or any(layer['descriptor'] != item['descriptor']
                        or layer['method'] != item['method'] for layer in layer_rows):
                    raise ValueError('Model cueq candidate did not execute one uniform combination')
            elif item['status'] not in ('unavailable', 'oom', 'failed_equivalence'):
                raise ValueError('Invalid model cueq candidate terminal status')
            if candidate.get('error'):
                item['reason'] = public_reason(candidate['error'])
            if candidate.get('error_type'):
                item['error_type'] = public_reason(candidate['error_type'])
            evidence = candidate.get('equivalence_evidence') or {}
            checksum = evidence.get('sha256')
            expected = new['payload_source'].get('model_compact_' +
                ('naive' if item['method'] == 'naive' else 'fused') + '_equivalence_sha256')
            if checksum != expected or evidence.get('qualification', {}).get('status') != 'passed':
                raise ValueError('Model cueq candidate is not pinned to its uniform D3 evidence')
            item['equivalence_sha256'] = checksum
            candidates_public.append(item)
        passed = [row for row in candidates_public if row['status'] == 'passed']
        if passed:
            winner = min(passed, key=lambda row: row['measurement']['forward_backward']['median_ms'])
            choice = source.get('cueq_selected') or {}
            if any(choice.get(key) != winner[key] for key in ('descriptor', 'method', 'rotation')):
                raise ValueError('Model cueq selection is not the fastest whole-model candidate')
            if cases[key]['implementations']['cueq']['forward_backward'] != winner['measurement']['forward_backward']:
                raise ValueError('Model cueq table does not reuse the recorded winning timing')
            cases[key]['cueq_selected'] = {field: winner[field] for field in ('descriptor', 'method', 'rotation')}
        cases[key]['cueq_alternatives'] = candidates_public
        if source.get('prior_mixed_candidate'):
            previous = source['prior_mixed_candidate']
            cases[key]['other_layer_combination'] = {
                'status': previous['status'], 'reason': public_reason(previous['reason']),
                'measurement': implementation(previous['measurement'])}
    for row in cases.values():
        row['implementations'].setdefault('cueq', {'status': 'N/A'})
        audit = applicability['models'][row['model']]
        if not audit['applicable']:
            if 'eqv3' in row['implementations']:
                raise ValueError('An inapplicable EquiformerV3 model was benchmarked')
            row['implementations']['eqv3'] = {'status': 'N/A', 'reason': audit['reason']}
        elif 'eqv3' not in row['implementations']:
            raise ValueError('Applicable EquiformerV3 model column is missing')
    return {'schema': 'so2cuda-public-model-benchmarks-v1', 'hardware': 'NVIDIA H200',
            'precision': 'FP32', 'tf32': False, 'warmup': 3, 'iterations': 10,
            'old_columns_retimed': False, 'source_commits': commits(new),
            'old_columns_source_commits': commits(old), 'cases': list(cases.values()),
            'eqv3_applicability': applicability}


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


def operator_tables(report):
    names = ['eqv3', 'naive', 'cueq', 'so2cuda']
    lines = []
    for group, title in (('A', '形状'), ('B', '规模'), ('C', '一般 irreps'), ('D', '截断 m')):
        rows = [row for row in report['cases'] if group in row['grids']]
        if not rows:
            continue
        rows.sort(key=lambda row: (row['config'].get('lmax', 0),
                                  row['config'].get('channels', 0),
                                  row['config'].get('irreps_in', ''),
                                  row['config']['edges']))
        lines += [f'**{title}**', '', '| 配置 | 有向边 | ' + ' | '.join(
            LABELS[name] + (' ms' if name == 'so2cuda' else ' ms（时间比）') for name in names) + ' |',
            '|---|---:|' + '---:|' * len(names)]
        for row in rows:
            accelerated = row['implementations']['so2cuda']
            cells = [value_cell(row['implementations'].get(name, {'status': 'N/A'}),
                                accelerated if name != 'so2cuda' else None) for name in names]
            lines.append(f'| {config_label(row["config"], group)} | {row["config"]["edges"]:,} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['<details>', '<summary>前向计时、四分位、峰值显存与 cuEquivariance 选择</summary>', '',
              '| 配置 | 有向边 | 实现 | 前向 ms（Q1–Q3） | 前向＋反向 ms（Q1–Q3） | 前向＋反向峰值 GiB |',
              '|---|---:|---|---:|---:|---:|']
    for row in report['cases']:
        detail_names = names + (['eqv3+compile'] if 'eqv3+compile' in row['implementations'] else [])
        for name in detail_names:
            impl = row['implementations'].get(name, {'status': 'N/A'})
            if impl['status'] != 'passed':
                continue
            fwd, both = impl['forward'], impl['forward_backward']
            peak = both.get('peak_allocated_bytes')
            memory = f'{peak / 2 ** 30:.2f}' if peak is not None else '—'
            lines.append(f'| {config_label(row["config"], row["grids"][0])} | {row["config"]["edges"]:,} | {LABELS[name]} | '
                f'{fwd["median_ms"]:.2f}（{fwd["q1_ms"]:.2f}–{fwd["q3_ms"]:.2f}） | '
                f'{both["median_ms"]:.2f}（{both["q1_ms"]:.2f}–{both["q3_ms"]:.2f}） | {memory} |')
    lines += ['', '| 配置 | 有向边 | cuEquivariance 描述符 | method | 旋转 | 版本 |', '|---|---:|---|---|---|---|']
    for row in report['cases']:
        choice = row['implementations'].get('cueq', {})
        if choice.get('status') == 'passed':
            lines.append(f'| {config_label(row["config"], row["grids"][0])} | {row["config"]["edges"]:,} | '
                         f'`{choice["descriptor"]}` | `{choice["method"]}` | `{choice["rotation"]}` | {choice["version"]} |')
    lines += ['', '全部候选的计时、数值结果与报错见 [算子 JSON](docs/benchmarks/OP_SPEED_H200.json)。', '', '</details>', '', ablation_table(report)]
    lines += ['', '<details>', '<summary>SO2CUDA 公开接口选择</summary>', '',
              '| 配置 | 有向边 | 所选接口 | forward_mode |', '|---|---:|---|---|']
    for row in report['cases']:
        choice = row['implementations']['so2cuda']
        if choice['status'] != 'passed':
            lines.append(f'| {config_label(row["config"], row["grids"][0])} | {row["config"]["edges"]:,} | OOM | — |')
            continue
        lines.append(f'| {config_label(row["config"], row["grids"][0])} | {row["config"]["edges"]:,} | '
                     f'`{choice["api"]}` | `{choice.get("forward_mode", "接口默认")}` |')
    lines += ['', '三个候选的全部计时与报错保留在算子 JSON 中。', '', '</details>']
    return '\n'.join(lines)


def ablation_table(report):
    lines = ['<details>', '<summary>显式旋转与 indexed sandwich 的消融</summary>', '',
        '中间变体用完整 Wigner 矩阵的 `torch.bmm` 完成旋转与逆旋转，正 m 的线性变换使用与'
        '默认 `dense_pairs` 路线相同的公开 `grouped_gemm_multi`，m=0 使用 `F.linear`。'
        '它通过完整 Wigner 旋转、一次 split 与 cat 组织逐 m 线性层，不调用 indexed sandwich 的 CUDA pack／scatter。', '',
        '依次比较 EquiformerV3 原版、中间变体与 SO2CUDA 的默认 `dense_pairs` 路线。分组 GEMM 的替换和 '
        'indexed sandwich 的打包／回写分别体现在两步差异中；三者的几何存储与布局也不同。'
        '峰值是几何准备完成后的计算峰值分配显存，包含预先计算的几何。', '',
        '| 配置 | 有向边 | 实现 | 前向 ms | 前向＋反向 ms | 前向峰值 GiB | 前向＋反向峰值 GiB |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in report['cases']:
        if not ablation_shape(row['config']):
            continue
        for name in ('eqv3', 'explicit_gemm', 'so2cuda_default', 'so2cuda_selected'):
            value = (row['ablation_so2cuda_default'] if name == 'so2cuda_default' else
                     row['implementations']['so2cuda'] if name == 'so2cuda_selected' else row['implementations'][name])
            if value['status'] == 'passed':
                fwd, both = value['forward'], value['forward_backward']
                cells = [f'{fwd["median_ms"]:.2f}', f'{both["median_ms"]:.2f}',
                         f'{fwd["peak_allocated_bytes"] / 2 ** 30:.2f}',
                         f'{both["peak_allocated_bytes"] / 2 ** 30:.2f}']
            else:
                cells = [('OOM' if value['status'] == 'oom' else 'N/A')] * 4
            lines.append(f'| {config_label(row["config"], row["grids"][0])} | {row["config"]["edges"]:,} | '
                         + ({'so2cuda_default': 'SO2CUDA dense_pairs 默认',
                             'so2cuda_selected': 'SO2CUDA 所选候选'}.get(name, LABELS.get(name, name)))
                         + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Q1/Q3、峰值 reserved 显存和等价性见 [算子 JSON](docs/benchmarks/OP_SPEED_H200.json)。', '', '</details>']
    return '\n'.join(lines)


def model_table(report):
    names = ['eqv3', 'naive', 'cueq', 'so2cuda']
    lines = ['| 模型 | 有向边 | ' + ' | '.join(LABELS[name] + ' ms' for name in names) + ' | 我们的纯 PyTorch／SO2CUDA |',
             '|---|---:|' + '---:|' * (len(names) + 1)]
    for row in report['cases']:
        impls = row['implementations']
        cells = [value_cell(impls.get(name, {'status': 'N/A'})) for name in names]
        ratio = (f'{impls["naive"]["forward_backward"]["median_ms"] / impls["so2cuda"]["forward_backward"]["median_ms"]:.2f}×'
                 if impls['naive']['status'] == impls['so2cuda']['status'] == 'passed' else '—')
        label = 'UniTB-dense' if row['model'] == 'dense' else 'UniTB'
        lines.append(f'| {label} | {row["edges"]:,} | ' + ' | '.join(cells + [ratio]) + ' |')
    return '\n'.join(lines)


def factual_operator_summary(report):
    names = {'eqv3': '与 EquiformerV3 原版相比', 'naive': '与我们的纯 PyTorch 实现相比', 'cueq': '与 cuEquivariance 相比'}
    lines = []
    for name in ('eqv3', 'naive', 'cueq'):
        pairs = [(row['implementations'][name]['forward_backward']['median_ms'],
                  row['implementations']['so2cuda']['forward_backward']['median_ms'])
                 for row in report['cases'] if row['implementations'].get(name, {}).get('status') == 'passed'
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
        lines.append(f'- {names[name]}：{verdict}，对方用时为 SO2CUDA 的 {span} 倍。')
    return '\n'.join(lines)


def model_choices_table(report):
    lines = ['<details>', '<summary>模型级 cuEquivariance 候选与选择</summary>', '',
             '| 模型 | 有向边 | 描述符 | method | 前向＋反向 ms | 状态 |', '|---|---:|---|---|---:|---|']
    for row in report['cases']:
        for candidate in row.get('cueq_alternatives', []):
            chosen = row.get('cueq_selected', {}).get('method') == candidate['method']
            timing = (f'{candidate["measurement"]["forward_backward"]["median_ms"]:.2f}'
                      if candidate['status'] == 'passed' else '—')
            label = 'UniTB-dense' if row['model'] == 'dense' else 'UniTB'
            state = '所选' if chosen else candidate['status']
            lines.append(f'| {label} | {row["edges"]:,} | `{candidate["descriptor"]}` | '
                         f'`{candidate["method"]}` | {timing} | {state} |')
    lines += ['', '每个边数分别用整模型比较全层统一的 `escn_tp_compact` 的 `naive` 与 `fused_tp`，'
              '所有六个 SO(2) 层使用同一组合，选择等价性通过且前向＋反向中位数最小的候选。', '', '</details>']
    return '\n'.join(lines)


def model_context(readme):
    if CONTEXT_BEGIN in readme and CONTEXT_END in readme:
        text = readme.split(CONTEXT_BEGIN, 1)[1].split(CONTEXT_END, 1)[0].strip()
    else:
        heading = '## UniTB 与 UniTB-dense 加速测试'
        if heading not in readme:
            raise ValueError('Existing model benchmark context is missing')
        text = readme.split(heading, 1)[1].split('## 构建与运行设置', 1)[0].split('### H200 实测', 1)[0].strip()
    text = text.replace(CLARIFICATION, '').strip()
    text = text.replace('朴素 PyTorch 基线', '我们的纯 PyTorch 实现').replace('朴素 PyTorch', '我们的纯 PyTorch')
    text = text.replace('**基线**在', '**我们的纯 PyTorch 实现**在')
    text = text.replace('基线调用 SO2CUDA', '我们的纯 PyTorch 实现调用 SO2CUDA')
    for sentence in ('我们的纯 PyTorch 实现调用 SO2CUDA 或加速路线未进入对应入口时会报错。',):
        text = text.replace(sentence, '若我们的纯 PyTorch 实现调用了 SO2CUDA，或加速路线没有进入对应入口，脚本会报错。')
    if CLARIFICATION not in text:
        text += '\n\n' + CLARIFICATION
    return text


def render(operator, model, equiv_op, equiv_model, context):
    versions = sorted({value['version'] for row in operator['cases']
        for name, value in row['implementations'].items() if name == 'cueq' and value.get('version')})
    version = ' / '.join(versions)
    if not version:
        raise ValueError('Measured cuEquivariance version is missing')
    warmup = min(row['warmup'] for row in operator['cases'])
    iterations = min(row['iterations'] for row in operator['cases'])
    methods = sorted({row['implementations']['cueq'].get('method') for row in operator['cases']
                      if row['implementations'].get('cueq', {}).get('status') == 'passed'})
    rotations = sorted({row['implementations']['cueq'].get('rotation') for row in operator['cases']
                        if row['implementations'].get('cueq', {}).get('status') == 'passed'})
    descriptors = sorted({row['implementations']['cueq']['descriptor'] for row in operator['cases']
                          if row['implementations'].get('cueq', {}).get('status') == 'passed'})
    so2_apis = sorted({row['implementations']['so2cuda']['api'] for row in operator['cases']
                      if row['implementations']['so2cuda']['status'] == 'passed'})
    if not equiv_op['all_passed'] or not equiv_model['all_passed']:
        raise ValueError('Numerical equivalence evidence failed')
    lines = [BEGIN, '## 性能测试', '', '### 1. SO(2) 张量积算子', '',
        '边上的 SO(2) 卷积先将输入旋转到边的主轴，做按 |m| 分组的共享线性变换，再旋转回原坐标：', '',
        r'$$y_e=D(R_e)^\top\,\mathcal{L}_W\!\left(D(R_e)x_e\right).$$', '',
        '`m=0` 使用实线性变换，`m>0` 使用 2×2 复结构；边之间共享权重，算子测试不含径向调制。', '',
        'SO2CUDA 的默认 `dense_pairs` 路线使用按索引打包的 Wigner sandwich 和分组 GEMM，'
        '再将输出回写到特征布局；非 MoE 的 `true_dense_pairs` 使用同一类 pack／scatter 和 `F.linear`。'
        'EquiformerV3 原版与我们的纯 PyTorch 实现分别显式旋转、计算线性层和逆旋转，均不包含这套 CUDA pack／scatter。', '',
        '- SO2CUDA：公开 `dense_pairs` 默认、`dense_pairs(forward_mode="indexed_sandwich_multi_grouped")` '
        '与 `true_dense_pairs` 三个候选；每个配置选取数值正确且前向＋反向最快的接口。'
        f'本表所选接口为 {" / ".join(f"`{name}`" for name in so2_apis)}，逐配置选择见下方详情。',
        '- 我们自己的纯 PyTorch 实现：按 DeePTB 上游 `SO2_Linear` 写法实现的算子，见 '
        '[operator_baselines.py](examples/operator_baselines.py)。',
        '- 朴素 PyTorch SO(2) 基线采用 EquiformerV3 原版：[固定提交](https://github.com/atomicarchitects/equiformer_v3/tree/'
        'a7300c58df683dc99cb48027d5bfd4c887486c48) 的 `SO3Rotation` ＋ `SO2Linear`（eager，源码不改）；'
        '`fc_m0` 的 bias 置零。',
        f'- cuEquivariance {version}：`SO3` irreps 的 {" / ".join(f"`{name}`" for name in descriptors)} 描述符由 `SegmentedPolynomial` 执行。'
        f'所选 method 为 {" / ".join(f"`{name}`" for name in methods)}，旋转方式为 '
        f'{" / ".join(f"`{name}`" for name in rotations)}；每个配置选取数值正确且前向＋反向最快的组合。', '',
        f'NVIDIA H200，计时前后核验独占，严格 FP32、关闭 TF32；几何量按各实现的格式预先计算。'
        '各实现使用各自原生特征布局，布局转换在计时区外；峰值显存只保留当前实现的输入、参数和几何量。'
        f'每路预热 {warmup} 次、计时 {iterations} 次。下表为前向＋输入和全部权重反向的 ms 中位数；'
        '括号内为该实现时间 ÷ SO2CUDA 时间：大于 1 表示 SO2CUDA 更快，小于 1 表示该实现更快。', '',
        factual_operator_summary(operator), '', operator_tables(operator), '',
        '输出、输入梯度、映射回共同参数的权重梯度及整体旋转等变检查，均通过 FP32 的绝对误差／相对 L2 联合判据；'
        '近零参考量使用绝对误差判据。逐项误差和判据见数值证据 JSON。', '',
        '各实现的参数映射与计时约定见 [docs/operator-benchmark.md](docs/operator-benchmark.md)；'
        '旋转与逐 m 线性的源码核查见 [docs/rotation-implementation-audit.md](docs/rotation-implementation-audit.md)。', '',
        '一般 irreps 表中的 EquiformerV3 为 N/A：原实现要求各 ℓ 的通道数一致；这些行的时间比相对我们的纯 PyTorch 和 '
        'cuEquivariance 报告。OOM 按原配置记录。', '',
        '复现算子测试（在本仓库根目录）：', '', '```bash',
        f'pip install cuequivariance-torch=={versions[0]}',
        'git clone https://github.com/atomicarchitects/equiformer_v3.git && git -C equiformer_v3 checkout a7300c58df683dc99cb48027d5bfd4c887486c48',
        'python examples/so2_operator_speed_test.py --impl all --lmax 6 --channels 128 --edges 50000 --warmup 5 --iterations 20 --eqv3-root equiformer_v3 --json operator.json',
        '```', '', '### 2. UniTB 模型', '', CONTEXT_BEGIN, context, CONTEXT_END, '',
        'NVIDIA H200，严格 FP32、计时前后核验独占；每路预热 3 次、计时 10 次。'
        '前向＋反向中位数沿用相同模型与计时口径，我们的纯 PyTorch 与 SO2CUDA 列使用既有测量，新增列单独实测。', '',
        model_table(model), '', model_choices_table(model), '',
        *[('UniTB-dense' if name == 'dense' else 'UniTB') + ' 的 EquiformerV3 为 N/A：' + row['reason']
          for name, row in model['eqv3_applicability']['models'].items() if not row['applicable']], '',
        '训练输出、推理输出和全部参数梯度均有限，并通过绝对误差／相对 L2 联合判据；'
        '近零梯度使用绝对误差判据。', '',
        'cuEquivariance 路线以标准子模块替换 SO(2) 算子，路由器、电荷头和其余模型层共享。'
        '每次前向共享一次 Wigner 构建，旋转用分块 bmm，再按描述符的原生布局执行线性层。'
        'N/A 表示该路线未提供忠实的对应实现；合成模型结果不代表物理精度或真实训练吞吐。', '',
        '完整计时、四分位与误差见 [模型 JSON](docs/benchmarks/MODEL_SPEED_H200.json)、'
        '[算子数值证据](docs/benchmarks/EQUIV_OP_L40S.json) 和 '
        '[模型数值证据](docs/benchmarks/EQUIV_MODEL_L40S.json)。', END]
    return '\n'.join(lines) + '\n'


def replace_section(readme, section):
    if BEGIN in readme and END in readme:
        start = readme.index(BEGIN)
        finish = readme.index(END, start) + len(END)
        return readme[:start] + section.rstrip() + readme[finish:]
    start = readme.index('## UniTB 与 UniTB-dense 加速测试')
    finish = readme.index('## 构建与运行设置', start)
    return readme[:start] + section + '\n' + readme[finish:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('operator', 'model', 'old-model', 'equiv-op', 'equiv-model', 'model-eqv3'):
        parser.add_argument('--' + name + '-json', type=Path, required=True)
    parser.add_argument('--equiv-model-fused-json', type=Path,
                        help='Explicit uniform compact/fused_tp D3 proof, when candidate model timings are supplied')
    parser.add_argument('--equiv-op-candidates-json', type=Path,
                        help='Fresh numerical and dispatch evidence for all three SO2CUDA public APIs')
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--readme', type=Path, default=root / 'README.md')
    parser.add_argument('--docs-dir', type=Path, default=root / 'docs/benchmarks')
    parser.add_argument('--write-readme', action='store_true', help='Apply the generated section to README')
    args = parser.parse_args()
    raw_operator, raw_model = load(args.operator_json), load(args.model_json)
    operator = operator_evidence(raw_operator)
    raw_applicability = load(args.model_eqv3_json)
    applicability = model_applicability(raw_applicability)
    if raw_operator.get('matrix', {}).get('model_eqv3_applicability') != raw_applicability:
        raise ValueError('Measured matrix differs from EquiformerV3 applicability receipt')
    model = model_evidence(load(args.old_model_json), raw_model, applicability)
    raw_equiv_op, raw_equiv_model = load(args.equiv_op_json), load(args.equiv_model_json)
    if raw_equiv_op.get('status') != 'passed' or raw_equiv_model.get('status') != 'passed':
        raise ValueError('Complete passed operator and model equivalence receipts are required')
    equiv_op, equiv_model = metrics(raw_equiv_op), metrics(raw_equiv_model)
    equiv_op['optional_candidates'] = selection_evidence(raw_equiv_op, 'operator')
    equiv_model['method_selections'] = selection_evidence(raw_equiv_model, 'model')
    inputs = {'operator': digest(args.operator_json), 'model': digest(args.model_json),
              'old_model': digest(args.old_model_json), 'equiv_op': digest(args.equiv_op_json),
              'equiv_model': digest(args.equiv_model_json), 'model_eqv3': digest(args.model_eqv3_json)}
    if args.equiv_model_fused_json:
        raw_fused = load(args.equiv_model_fused_json)
        if raw_fused.get('status') != 'passed':
            raise ValueError('Uniform compact/fused_tp equivalence must pass')
        combined = {'models': [raw_equiv_model, raw_fused]}
        equiv_model = metrics(combined)
        equiv_model['method_selections'] = selection_evidence(raw_equiv_model, 'model') + selection_evidence(raw_fused, 'model')
        inputs['equiv_model_fused'] = digest(args.equiv_model_fused_json)
    if args.equiv_op_candidates_json:
        raw_candidates = load(args.equiv_op_candidates_json)
        if raw_candidates.get('status') != 'passed':
            raise ValueError('All SO2CUDA public API candidate numerical checks must pass')
        equiv_op = metrics({'original': raw_equiv_op, 'so2cuda_candidates': raw_candidates})
        equiv_op['optional_candidates'] = selection_evidence(raw_equiv_op, 'operator')
        equiv_op['public_api_candidates'] = []
        for section in ('small', 'large'):
            for value in raw_candidates.get(section, {}).values():
                if value.get('candidate') not in SO2CUDA_CANDIDATES or value.get('status') != 'passed':
                    raise ValueError('SO2CUDA public API numerical candidate failed')
                equiv_op['public_api_candidates'].append({'candidate': value['candidate'], 'scope': section,
                    'edges': number(value['edges']), 'equivalence': metrics(value['result'])})
        dispatch = raw_candidates.get('dispatch', {})
        if set(dispatch) != SO2CUDA_CANDIDATES or any(row.get('status') != 'passed' for row in dispatch.values()):
            raise ValueError('SO2CUDA candidate dispatch checks did not all pass')
        equiv_op['dispatch_verified'] = {key: True for key in sorted(dispatch)}
        inputs['equiv_op_candidates'] = digest(args.equiv_op_candidates_json)
    frozen_source(raw_operator, raw_model, inputs)
    readme = args.readme.read_text(encoding='utf-8')
    section = render(operator, model, equiv_op, equiv_model, model_context(readme))
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (('OP_SPEED_H200', operator), ('MODEL_SPEED_H200', model),
                        ('EQUIV_OP_L40S', equiv_op), ('EQUIV_MODEL_L40S', equiv_model),
                        ('MODEL_EQV3_APPLICABILITY', applicability)):
        (args.docs_dir / (name + '.json')).write_text(
            json.dumps({**value, 'input_sha256': inputs}, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    (args.docs_dir / 'README_SECTION.md').write_text(section, encoding='utf-8')
    if args.write_readme:
        args.readme.write_text(replace_section(readme, section), encoding='utf-8')
    print(json.dumps({'public_json_files': 5, 'readme_updated': args.write_readme}, ensure_ascii=False))


if __name__ == '__main__':
    main()
