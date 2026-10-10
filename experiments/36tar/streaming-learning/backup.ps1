param([switch]$CleanStaging)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$backupRoot = 'C:\Users\21195\Desktop\大模型学习\流式视频模型架构学习_交接备份'
$releaseId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stageRoot = Join-Path $backupRoot ('release-' + $releaseId)
$archivePath = Join-Path $backupRoot ('streaming-design-' + $releaseId + '.zip')
$allowList = @('AGENTS.md', '实验总纲.md', '架构设计.md', '协作与集群操作.md', 'configs/prototype.json', 'backup.ps1')
$allowList += '合作仓库架构与计划解读.md'
$allowList += @('语言计算块交错扩展方案.md', '语言计算块交错扩展结果.md',
    'scripts/run_chunk_interleaved.py', 'scripts/verify_chunk_interleaved.py',
    'code/qwen35-streaming-s4/scripts/chunk_interleaved.py',
    'runs/s4b_chunk_interleaved_delivery_check_20261010_01/verification_script_snapshot.py',
    'runs/s4b_chunk_interleaved_delivery_check_20261010_01/audit_failure.txt',
    'runs/s4b_chunk_interleaved_gpu_20261010_01/resource_release_check.json')
foreach ($name in @('verification.json', 'source_tree.json', 'source_snapshots.json', 'tracked_changes.patch',
    'cleanup_verification.json', 'verification_script_snapshot.py', 'document_verification.json', 'backup_verification.json')) {
    $relativeAudit = 'runs/s4b_chunk_interleaved_delivery_check_20261010_02/' + $name
    if (Test-Path -LiteralPath (Join-Path $projectRoot $relativeAudit) -PathType Leaf) { $allowList += $relativeAudit }
}
$allowList += @('流式视频架构学习复盘与未解决问题.md',
    'runs/s6_recap_check_20261010_01/verification.json',
    'runs/s6_recap_check_20261010_01/backup_verification.json')
$allowList += @('问题优先级与下一步实验建议.md',
    'runs/s6_priority_check_20261010_01/verification.json')
$allowList += @('视觉首次差异定位方案.md', '视觉首次差异定位结果.md',
    '视觉QKV同输入重放方案.md', '视觉QKV同输入重放结果.md',
    '固定时间单元视觉参考方案.md', '固定时间单元视觉参考结果.md',
    '固定视觉在线回归方案.md', '固定视觉在线回归结果.md', '视觉一致性探索复盘.md',
    'scripts/run_vision_trace.py', 'scripts/run_qkv_replay.py', 'scripts/run_vision_fixed.py',
    'scripts/run_vision_fixed_online.py', 'scripts/verify_vision_exploration.py',
    'code/qwen35-streaming-s4/scripts/vision_trace.py', 'code/qwen35-streaming-s4/scripts/qkv_replay.py',
    'code/qwen35-streaming-s4/scripts/vision_fixed_units.py', 'code/qwen35-streaming-s4/scripts/vision_fixed_control.py',
    'code/qwen35-streaming-s4/scripts/vision_fixed_online.py',
    'runs/s4b_vision_fixed_online_preflight_20261010_01/plan_snapshot.md',
    'runs/s4b_vision_exploration_delivery_check_20261010_01/verification_script_snapshot.py',
    'runs/s4b_vision_exploration_delivery_check_20261010_01/resource_release_check.json',
    'runs/s4b_vision_exploration_delivery_check_20261010_02/verification_script_snapshot.py',
    'runs/s4b_vision_exploration_delivery_check_20261010_02/document_verification.json',
    'runs/s4b_vision_exploration_delivery_check_20261010_02/backup_verification.json')
foreach ($audit in @('s4b_vision_exploration_delivery_check_20261010_01', 's4b_vision_exploration_delivery_check_20261010_02')) {
    foreach ($name in @('verification.json', 'source_tree.json', 'source_snapshots.json', 'tracked_changes.patch', 'cleanup_verification.json')) {
        $allowList += 'runs/' + $audit + '/' + $name
    }
}
$allowList += '流式模型定向调研与借鉴路线.md'
$allowList += @('扩展视觉一致性验收方案.md', '扩展视觉一致性验收结果.md',
    '分阶段耗时验收方案.md', '分阶段耗时验收结果.md', '语言计算块对照方案.md',
    '语言计算块对照结果.md', '视觉扩展与性能探索复盘.md',
    'scripts/run_vision_extension.py', 'scripts/run_stage_timing.py', 'scripts/run_chunk_probe.py',
    'scripts/verify_vision_extension.py', 'code/qwen35-streaming-s4/scripts/vision_expand.py',
    'code/qwen35-streaming-s4/scripts/stage_timing.py', 'code/qwen35-streaming-s4/scripts/chunk_size_probe.py',
    'runs/s4b_stage_timing_gpu_20261010_01/timing_partial.json')
foreach ($name in @('verification.json', 'source_tree.json', 'source_snapshots.json', 'tracked_changes.patch',
    'cleanup_verification.json', 'verification_script_snapshot.py', 'resource_release_check.json',
    'document_verification.json', 'backup_verification.json')) {
    $relativeAudit = 'runs/s4b_vision_extension_delivery_check_20261010_01/' + $name
    if (Test-Path -LiteralPath (Join-Path $projectRoot $relativeAudit) -PathType Leaf) { $allowList += $relativeAudit }
}
$snapshotFiles = @('SNAPSHOT.json', '.gitattributes', '.gitignore', 'README.md', 'THIRD_PARTY_NOTICES.md', 'requirements.txt', 'qwen35_local/__init__.py', 'qwen35_local/configuration_qwen3_5.py', 'qwen35_local/modeling_qwen3_5.py', 'scripts/__init__.py', 'scripts/smoke_test.py', 'tests/__init__.py', 'tests/test_local_model.py', 'third_party/transformers-LICENSE', 'third_party/upstream.json')
$allowList += $snapshotFiles | ForEach-Object { 'reference/qwen35-streaming/' + $_ }
$allowList += $snapshotFiles | ForEach-Object { 'code/qwen35-streaming/' + $_ }
$allowList += @('基线实验与机制引入计划.md', '基线结果与下一步.md', '基线复现说明.md',
    'scripts/download_baseline_model.ps1', 'scripts/run_baseline.py',
    'code/qwen35-streaming/scripts/baseline.py', 'code/qwen35-streaming/requirements-baseline.lock.txt',
    'datasets/baseline_synthetic/manifest.json', 'models/Qwen3.5-0.8B/DOWNLOAD_MANIFEST.json',
    'models/Qwen3.5-0.8B/DOWNLOAD_METADATA.json')
$allowList += @('在线滑窗设计与验收.md', '在线滑窗结果与复现.md', 'scripts/run_window.py',
    'scripts/build_window_visual.py', 'scripts/window_replay.fragment.html',
    'code/qwen35-streaming/streaming/__init__.py', 'code/qwen35-streaming/streaming/window.py',
    'code/qwen35-streaming/streaming/inputs.py', 'code/qwen35-streaming/scripts/window_experiment.py',
    'code/qwen35-streaming/tests/test_stream_window.py', 'datasets/window_synthetic/manifest.json',
    'runs/window_visual_check_20261009_01/visual_check.json', 'runs/window_visual_check_20261009_02/visual_check.json',
    'runs/window_delivery_check_20261009_01/verification.json')
$allowList += @('长运行与重置验收方案.md', '长运行与重置结果.md', 'scripts/run_stability.py',
    'scripts/run_timestamp_diagnostic.py', 'code/qwen35-streaming/scripts/timestamp_diagnostic.py',
    'code/qwen35-streaming/streaming/session.py', 'code/qwen35-streaming/tests/test_stream_session.py',
    'code/qwen35-streaming/scripts/stability_experiment.py', 'datasets/stability_synthetic/manifest.json',
    'runs/stability_delivery_check_20261009_01/verification.json')
$allowList += @('自然视频验收方案.md', '自然视频结果与复现.md',
    'scripts/download_natural_videos.py', 'scripts/prepare_natural_cases.py', 'scripts/run_natural.py',
    'code/qwen35-streaming/streaming/inference.py', 'code/qwen35-streaming/scripts/natural_experiment.py',
    'datasets/natural_commons_v1/manifest.json', 'datasets/natural_commons_v1/cases.json',
    'datasets/natural_commons_v1/ATTRIBUTION.md',
    'runs/natural_inspection_20261009_01/source_probe.json',
    'runs/natural_inspection_20261009_01/source_vfr_probe.json',
    'runs/natural_delivery_check_20261009_01/verification.json')
$allowList += @('混合状态续接验收方案.md', '混合状态续接结果与复现.md',
    'scripts/run_mixed_state.py', 'scripts/verify_mixed_state_delivery.py',
    'runs/s4_setup_20261009_01/setup.json',
    'runs/s4_delivery_check_20261009_01/verification.json',
    'runs/s4_delivery_check_20261009_01/source_tree.json',
    'runs/s4_delivery_check_20261009_01/git_review.json',
    'runs/s4_delivery_check_20261009_01/tracked_changes.patch',
    'runs/s4_precision_cpu_20261009_01/script_snapshot.py',
    'runs/s4_precision_cpu_20261009_02/script_snapshot.py')
$allowList += @('逐层精度诊断方案.md', '逐层精度诊断与递归参考结果.md',
    'scripts/run_precision_layers.py', 'scripts/verify_precision_delivery.py',
    'runs/s4_before_recurrent_20261009_01/modeling_snapshot.py',
    'runs/s4_before_recurrent_20261009_01/source.json',
    'runs/s4_before_recurrent_20261009_01/chunk_experiment_snapshot.py',
    'runs/s4_before_recurrent_20261009_01/chunk_weight_check_snapshot.py',
    'runs/s4_before_recurrent_20261009_01/prior_scripts.json',
    'runs/s4_recurrent_cpu_20261009_01/chunk_weight_check_snapshot.py',
    'runs/s4_layers_preflight_20261009_01/script_snapshot.py',
    'runs/s4_layers_preflight_20261009_02/script_snapshot.py',
    'runs/s4_layers_gpu_20261009_01/script_snapshot.py',
    'runs/s4_recurrence_cpu_20261009_01/script_snapshot.py',
    'runs/s4_precision_delivery_check_20261009_01/verification.json',
    'runs/s4_precision_delivery_check_20261009_01/source_tree.json',
    'runs/s4_precision_delivery_check_20261009_01/git_review.json',
    'runs/s4_precision_delivery_check_20261009_01/tracked_changes.patch')
$allowList += @('高精度整模型诊断方案.md', '高精度整模型诊断结果.md',
    'scripts/run_precision_fullmodel.py', 'scripts/verify_full_precision_delivery.py',
    'runs/s4_full_precision_preflight_20261009_01/script_snapshot.py',
    'runs/s4_full_precision_gpu_20261009_01/script_snapshot.py',
    'runs/s4_full_precision_delivery_check_20261009_01/verification.json',
    'runs/s4_full_precision_delivery_check_20261009_01/source_tree.json',
    'runs/s4_full_precision_delivery_check_20261009_01/tracked_changes.patch',
    'runs/s4_precision_delivery_check_20261009_02/verification.json',
    'runs/s4_precision_delivery_check_20261009_02/source_tree.json',
    'runs/s4_precision_delivery_check_20261009_02/git_review.json',
    'runs/s4_precision_delivery_check_20261009_02/tracked_changes.patch')
$allowList += @('FP32混合状态回归方案.md', 'FP32混合状态回归结果.md', '在线成对帧与问答协议方案.md',
    'scripts/run_fp32_regression.py', 'scripts/verify_fp32_regression_delivery.py', 'scripts/probe_online_protocol.py',
    'runs/s4_fp32_regression_preflight_20261009_01/script_snapshot.py',
    'runs/s4_fp32_regression_gpu_20261009_01/script_snapshot.py',
    'runs/s4b_protocol_cpu_20261009_01/script_snapshot.py',
    'runs/s4_fp32_regression_delivery_check_20261009_01/verification.json',
    'runs/s4_fp32_regression_delivery_check_20261009_01/source_tree.json',
    'runs/s4_fp32_regression_delivery_check_20261009_01/tracked_changes.patch',
    'runs/s4_fp32_regression_delivery_check_20261009_02/verification.json',
    'runs/s4_fp32_regression_delivery_check_20261009_02/source_tree.json',
    'runs/s4_fp32_regression_delivery_check_20261009_02/tracked_changes.patch')
$allowList += @('在线成对帧问答验收方案.md', '在线成对帧问答结果与复现.md', '在线问答失败定位与后续控制.md',
    'scripts/run_online_pairs.py', 'scripts/verify_online_delivery.py',
    'runs/s4b_online_preflight_20261009_01/script_snapshot.py',
    'runs/s4b_online_preflight_20261009_01/online_snapshot.py',
    'runs/s4b_online_preflight_20261009_01/tests_snapshot.py',
    'runs/s4b_online_gpu_20261009_01/script_snapshot.py',
    'runs/s4b_online_gpu_20261009_01/online_snapshot.py',
    'runs/s4b_online_gpu_20261009_01/tests_snapshot.py',
    'runs/s4b_online_preflight_20261009_02/script_snapshot.py',
    'runs/s4b_online_preflight_20261009_02/online_snapshot.py',
    'runs/s4b_online_preflight_20261009_02/tests_snapshot.py',
    'runs/s4b_online_gpu_20261009_01/frame_2.png', 'runs/s4b_online_gpu_20261009_01/frame_4.png',
    'runs/s4b_online_delivery_check_20261009_01/verification.json',
    'runs/s4b_online_delivery_check_20261009_01/source_tree.json',
    'runs/s4b_online_delivery_check_20261009_01/tracked_changes.patch')
$allowList += @('共享视觉特征控制方案.md', '共享视觉特征控制结果与复现.md',
    'scripts/run_shared_features.py', 'scripts/verify_shared_features_delivery.py',
    'code/qwen35-streaming-s4/scripts/shared_features_control.py',
    'runs/s4b_shared_preflight_20261009_01/script_snapshot.py',
    'runs/s4b_shared_preflight_20261009_01/launcher_snapshot.py',
    'runs/s4b_shared_preflight_20261009_02/script_snapshot.py',
    'runs/s4b_shared_gpu_20261009_01/script_snapshot.py',
    'runs/s4b_shared_delivery_check_20261009_01/verification.json',
    'runs/s4b_shared_delivery_check_20261009_01/source_tree.json',
    'runs/s4b_shared_delivery_check_20261009_01/tracked_changes.patch')
$allowList += @('同画面输入与能力控制方案.md', '同画面输入与能力控制结果.md',
    'scripts/run_protocol_capability.py', 'scripts/verify_protocol_capability_delivery.py',
    'code/qwen35-streaming-s4/scripts/protocol_capability_control.py',
    'runs/s4b_protocol_capability_preflight_20261009_01/script_snapshot.py',
    'runs/s4b_protocol_capability_gpu_20261009_01/script_snapshot.py',
    'runs/s4b_protocol_capability_delivery_check_20261009_01/verification.json',
    'runs/s4b_protocol_capability_delivery_check_20261009_01/source_tree.json',
    'runs/s4b_protocol_capability_delivery_check_20261009_01/tracked_changes.patch',
    'runs/s4b_protocol_capability_delivery_check_20261009_02/verification.json',
    'runs/s4b_protocol_capability_delivery_check_20261009_02/source_tree.json',
    'runs/s4b_protocol_capability_delivery_check_20261009_02/tracked_changes.patch')
$allowList += @('在线四帧主体公平对照方案.md', '在线四帧主体公平对照结果.md', '同步成对帧演示与触发设计.md',
    'scripts/run_four_frame_online.py', 'scripts/verify_four_frame_online_delivery.py',
    'code/qwen35-streaming-s4/scripts/four_frame_online.py',
    'runs/s4b_four_frame_preflight_20261010_01/script_snapshot.py',
    'runs/s4b_four_frame_gpu_20261010_01/script_snapshot.py',
    'runs/s4b_four_frame_preflight_20261010_02/script_snapshot.py',
    'runs/s4b_four_frame_gpu_20261010_02/script_snapshot.py',
    'runs/s4b_four_frame_delivery_check_20261010_01/script_snapshot.py',
    'runs/s4b_four_frame_delivery_check_20261010_01/audit_failure.txt',
    'runs/s4b_four_frame_delivery_check_20261010_02/verification.json',
    'runs/s4b_four_frame_delivery_check_20261010_02/source_tree.json',
    'runs/s4b_four_frame_delivery_check_20261010_02/tracked_changes.patch')
$allowList += $snapshotFiles | ForEach-Object { 'code/qwen35-streaming-s4/' + $_ }
$allowList += @('跨场景与早期画面对照方案.md', '跨场景与早期画面对照结果.md',
    'scripts/run_scene_pairs.py', 'scripts/verify_scene_pairs_delivery.py',
    'code/qwen35-streaming-s4/scripts/scene_pair_control.py',
    'runs/s4b_scene_fixture_20261010_01/fixture_sources.json',
    'runs/s4b_scene_fixture_20261010_01/contact_sheet.png',
    'runs/s4b_scene_pairs_preflight_20261010_01/script_snapshot.py',
    'runs/s4b_scene_pairs_gpu_20261010_01/script_snapshot.py',
    'runs/s4b_scene_pairs_preflight_20261010_01/events.jsonl',
    'runs/s4b_scene_pairs_gpu_20261010_01/events.jsonl',
    'runs/s4b_scene_pairs_delivery_check_20261010_01/verification.json',
    'runs/s4b_scene_pairs_delivery_check_20261010_01/backup_verification.json',
    'runs/s4b_scene_pairs_delivery_check_20261010_01/source_tree.json',
    'runs/s4b_scene_pairs_delivery_check_20261010_01/tracked_changes.patch')
$allowList += @('同步成对帧演示验收方案.md', '同步成对帧演示结果与复现.md',
    'scripts/run_pair_demo.py', 'scripts/verify_pair_demo_delivery.py',
    'code/qwen35-streaming-s4/streaming/demo.py',
    'code/qwen35-streaming-s4/tests/test_pair_demo.py',
    'code/qwen35-streaming-s4/scripts/pair_demo.py',
    'runs/s4b_pair_demo_preflight_20261010_01/script_snapshot.py',
    'runs/s4b_pair_demo_gpu_20261010_01/script_snapshot.py',
    'runs/s4b_pair_demo_preflight_20261010_01/events.jsonl',
    'runs/s4b_pair_demo_gpu_20261010_01/events.jsonl',
    'runs/s4b_pair_demo_delivery_check_20261010_01/verification.json',
    'runs/s4b_pair_demo_delivery_check_20261010_01/backup_verification.json',
    'runs/s4b_pair_demo_delivery_check_20261010_01/source_tree.json',
    'runs/s4b_pair_demo_delivery_check_20261010_01/tracked_changes.patch')
$s4Extras = @('S4_A.md', 'requirements-baseline.lock.txt',
    'scripts/baseline.py', 'scripts/window_experiment.py', 'scripts/stability_experiment.py',
    'scripts/timestamp_diagnostic.py', 'scripts/natural_experiment.py', 'scripts/chunk_experiment.py',
    'scripts/chunk_weight_check.py', 'scripts/precision_diagnostic_cpu.py',
    'scripts/precision_layers.py', 'scripts/precision_recurrence_cpu.py', 'scripts/precision_fullmodel.py',
    'scripts/fp32_regression.py', 'scripts/online_experiment.py',
    'streaming/__init__.py', 'streaming/window.py', 'streaming/inputs.py', 'streaming/session.py',
    'streaming/inference.py', 'streaming/hybrid.py', 'streaming/online.py',
    'tests/test_stream_window.py', 'tests/test_stream_session.py', 'tests/test_mixed_state.py',
    'tests/test_recurrent_reference.py', 'tests/test_online.py')
$allowList += $s4Extras | ForEach-Object { 'code/qwen35-streaming-s4/' + $_ }
$evidenceRuns = @('baseline_assets_20261009_01', 'baseline_cpu_20261009_01', 'baseline_cpu_20261009_02',
    'baseline_preflight_20261009_01', 'baseline_preflight_20261009_02', 'baseline_preflight_20261009_03',
    'baseline_gpu_20261009_01', 'baseline_gpu_20261009_02',
    'window_assets_20261009_01', 'window_cpu_20261009_01', 'window_cpu_20261009_02',
    'window_preflight_20261009_01', 'window_gpu_20261009_01',
    'stability_assets_20261009_01', 'stability_cpu_20261009_01', 'stability_cpu_20261009_02',
    'stability_preflight_20261009_01', 'stability_gpu_20261009_01',
    'timestamp_preflight_20261009_01', 'timestamp_gpu_20261009_01',
    'natural_preflight_20261009_01', 'natural_gpu_20261009_01',
    's4_before_20261009_01', 's4_cpu_20261009_01', 's4_preflight_20261009_01',
    's4_gpu_20261009_01', 's4_after_20261009_01',
    's4_precision_cpu_20261009_01', 's4_precision_cpu_20261009_02',
    's4_layers_preflight_20261009_01', 's4_layers_preflight_20261009_02', 's4_layers_gpu_20261009_01',
    's4_recurrence_cpu_20261009_01', 's4_recurrent_cpu_20261009_01', 's4_recurrent_cpu_20261009_02',
    's4_recurrent_preflight_20261009_01', 's4_recurrent_gpu_20261009_01',
    's4_full_precision_preflight_20261009_01', 's4_full_precision_gpu_20261009_01',
    's4_fp32_regression_preflight_20261009_01', 's4_fp32_regression_gpu_20261009_01',
    's4b_protocol_cpu_20261009_01', 's4b_online_preflight_20261009_01',
    's4b_online_gpu_20261009_01', 's4b_online_preflight_20261009_02',
    's4b_shared_preflight_20261009_01', 's4b_shared_preflight_20261009_02', 's4b_shared_gpu_20261009_01',
    's4b_protocol_capability_preflight_20261009_01', 's4b_protocol_capability_gpu_20261009_01',
    's4b_four_frame_preflight_20261010_01', 's4b_four_frame_gpu_20261010_01',
    's4b_four_frame_preflight_20261010_02', 's4b_four_frame_gpu_20261010_02')
$evidenceRuns += @('s4b_pair_demo_preflight_20261010_01', 's4b_pair_demo_gpu_20261010_01')
$evidenceRuns += @('s4b_scene_pairs_preflight_20261010_01', 's4b_scene_pairs_gpu_20261010_01')
$allowList += @('原模型早期主体对照方案.md', '原模型早期主体对照结果.md',
    'scripts/run_original_first.py', 'scripts/verify_original_first_delivery.py',
    'code/qwen35-streaming-s4/scripts/original_first_control.py',
    'runs/s4b_original_first_preflight_20261010_01/script_snapshot.py',
    'runs/s4b_original_first_preflight_20261010_02/script_snapshot.py',
    'runs/s4b_original_first_preflight_20261010_03/script_snapshot.py',
    'runs/s4b_original_first_gpu_20261010_01/script_snapshot.py',
    'runs/s4b_original_first_gpu_20261010_02/script_snapshot.py',
    'runs/s4b_original_first_gpu_20261010_03/script_snapshot.py',
    'runs/s4b_original_first_delivery_check_20261010_01/audit_failure.txt',
    'runs/s4b_original_first_delivery_check_20261010_02/verification.json',
    'runs/s4b_original_first_delivery_check_20261010_02/backup_verification.json',
    'runs/s4b_original_first_delivery_check_20261010_02/source_tree.json',
    'runs/s4b_original_first_delivery_check_20261010_02/tracked_changes.patch')
$evidenceRuns += @('s4b_original_first_preflight_20261010_01', 's4b_original_first_preflight_20261010_02',
    's4b_original_first_preflight_20261010_03', 's4b_original_first_gpu_20261010_01',
    's4b_original_first_gpu_20261010_02', 's4b_original_first_gpu_20261010_03')
$allowList += @('原模型早期四帧对照方案.md', '原模型早期四帧对照结果.md',
    'scripts/run_prefix_first.py', 'scripts/verify_prefix_first_delivery.py',
    'code/qwen35-streaming-s4/scripts/prefix_first_control.py',
    'runs/s4b_prefix_first_preflight_20261010_01/script_snapshot.py',
    'runs/s4b_prefix_first_gpu_20261010_01/script_snapshot.py',
    'runs/s4b_prefix_first_delivery_check_20261010_01/verification.json',
    'runs/s4b_prefix_first_delivery_check_20261010_01/backup_verification.json',
    'runs/s4b_prefix_first_delivery_check_20261010_01/source_tree.json',
    'runs/s4b_prefix_first_delivery_check_20261010_01/tracked_changes.patch')
$evidenceRuns += @('s4b_prefix_first_preflight_20261010_01', 's4b_prefix_first_gpu_20261010_01')
$allowList += @('原模型六帧等长末尾对照方案.md', '原模型六帧等长末尾对照结果.md',
    'scripts/run_equal_tail.py', 'scripts/verify_equal_tail_delivery.py',
    'code/qwen35-streaming-s4/scripts/equal_tail_control.py',
    'runs/s4b_equal_tail_preflight_20261010_01/script_snapshot.py',
    'runs/s4b_equal_tail_gpu_20261010_01/script_snapshot.py',
    'runs/s4b_equal_tail_delivery_check_20261010_01/verification.json',
    'runs/s4b_equal_tail_delivery_check_20261010_01/backup_verification.json',
    'runs/s4b_equal_tail_delivery_check_20261010_01/source_tree.json',
    'runs/s4b_equal_tail_delivery_check_20261010_01/tracked_changes.patch')
$evidenceRuns += @('s4b_equal_tail_preflight_20261010_01', 's4b_equal_tail_gpu_20261010_01')
$visionRuns = @('s4b_vision_trace_preflight_20261010_01', 's4b_vision_trace_gpu_20261010_01',
    's4b_qkv_replay_preflight_20261010_01', 's4b_qkv_replay_gpu_20261010_01',
    's4b_vision_fixed_preflight_20261010_01', 's4b_vision_fixed_preflight_20261010_02', 's4b_vision_fixed_gpu_20261010_01',
    's4b_vision_fixed_online_preflight_20261010_01', 's4b_vision_fixed_online_preflight_20261010_02', 's4b_vision_fixed_online_gpu_20261010_01')
$evidenceRuns += $visionRuns
$extensionRuns = @('s4b_vision_expand_preflight_20261010_01', 's4b_vision_expand_gpu_20261010_01',
    's4b_stage_timing_preflight_20261010_01', 's4b_stage_timing_gpu_20261010_01',
    's4b_stage_timing_preflight_20261010_02', 's4b_stage_timing_gpu_20261010_02',
    's4b_chunk_probe_preflight_20261010_01', 's4b_chunk_probe_gpu_20261010_01', 's4b_chunk_probe_gpu_20261010_02')
$evidenceRuns += $extensionRuns
$interleavedRuns = @('s4b_chunk_interleaved_preflight_20261010_01', 's4b_chunk_interleaved_gpu_20261010_01')
$evidenceRuns += $interleavedRuns
foreach ($run in $interleavedRuns) {
    foreach ($name in @('script_snapshot.py', 'launcher_snapshot.py', 'plan_snapshot.md')) {
        $allowList += 'runs/' + $run + '/' + $name
    }
}
foreach ($run in $extensionRuns) {
    foreach ($name in @('script_snapshot.py', 'launcher_snapshot.py', 'plan_snapshot.md')) {
        $allowList += 'runs/' + $run + '/' + $name
    }
}
foreach ($run in $visionRuns) {
    $allowList += 'runs/' + $run + '/script_snapshot.py'
    $allowList += 'runs/' + $run + '/launcher_snapshot.py'
}
$evidenceNames = @('interleaved_summary.json', 'expansion_summary.json', 'timing_summary.json', 'chunk_probe_summary.json', 'runtime_cleanup.json',
    'vision_trace_summary.json', 'qkv_replay_summary.json', 'vision_fixed_summary.json',
    'equal_tail_summary.json', 'prefix_first_summary.json', 'original_first_summary.json', 'scene_pair_summary.json', 'fixture_summary.json', 'pair_demo_summary.json', 'cleanup.json', 'status.json', 'provenance.json', 'arguments.json', 'assets_summary.json',
    'cpu_summary.json', 'preflight_summary.json', 'gpu_summary.json', 'diagnostic_summary.json',
    'precision_summary.json', 'precision_layers_summary.json', 'recurrence_summary.json', 'script_snapshot.py.json',
    'full_precision_summary.json', 'resource_precheck.json', 'fp32_regression_summary.json', 'protocol_summary.json', 'online_summary.json', 'shared_features_summary.json', 'protocol_capability_summary.json', 'four_frame_online_summary.json',
    'traceback.txt', 'unittest.log', 'watchdog_status.json')
foreach ($run in $evidenceRuns) {
    foreach ($name in $evidenceNames) {
        $relative = 'runs/' + $run + '/' + $name
        if (Test-Path -LiteralPath (Join-Path $projectRoot $relative) -PathType Leaf) { $allowList += $relative }
    }
    $console = 'runs/' + $run + '.console.log'
    if (Test-Path -LiteralPath (Join-Path $projectRoot $console) -PathType Leaf) { $allowList += $console }
}
if ((Test-Path -LiteralPath $stageRoot) -or (Test-Path -LiteralPath $archivePath)) { throw 'Fresh backup release path required' }
New-Item -ItemType Directory -Path $stageRoot | Out-Null
$totalBytes = 0
$manifest = foreach ($relativePath in $allowList) {
    $sourcePath = Join-Path $projectRoot $relativePath
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) { throw "Missing required file: $relativePath" }
    $fileBytes = (Get-Item -LiteralPath $sourcePath).Length
    if ($fileBytes -gt 1MB) { throw "Backup whitelist file exceeds small-evidence limit: $relativePath" }
    $totalBytes += $fileBytes
    if ($totalBytes -gt 10MB) { throw 'Backup exceeded total small-file budget' }
    $targetPath = Join-Path $stageRoot $relativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
    Copy-Item -LiteralPath $sourcePath -Destination $targetPath
    [PSCustomObject]@{Path=$relativePath; SHA256=(Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash}
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stageRoot 'MANIFEST.sha256.json') -Encoding UTF8
Compress-Archive -Path (Join-Path $stageRoot '*') -DestinationPath $archivePath
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    foreach ($item in $manifest) {
        $entry = $zip.GetEntry($item.Path.Replace('\', '/'))
        if ($null -eq $entry) { throw "Archive missing file: $($item.Path)" }
        $entryStream = $entry.Open()
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try { $actualHash = ([BitConverter]::ToString($sha.ComputeHash($entryStream))).Replace('-', '') }
        finally { $entryStream.Dispose(); $sha.Dispose() }
        if ($actualHash -ne $item.SHA256) { throw "Archive hash mismatch: $($item.Path)" }
    }
} finally { $zip.Dispose() }
# Default retains an inspectable release. Optional cleanup only removes this verified, newly created staging copy.
$stagingRemoved = $false
if ($CleanStaging) {
    $resolvedBackup = (Resolve-Path -LiteralPath $backupRoot).Path.TrimEnd([IO.Path]::DirectorySeparatorChar)
    $resolvedStage = (Resolve-Path -LiteralPath $stageRoot).Path
    $prefix = $resolvedBackup + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedStage.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolvedStage) -ne ('release-' + $releaseId)) { throw 'Unsafe staging cleanup target' }
    $links = @(Get-ChildItem -LiteralPath $resolvedStage -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
    if ($links.Count -gt 0) { throw 'Refusing cleanup through reparse points' }
    Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    $stagingRemoved = -not (Test-Path -LiteralPath $resolvedStage)
}
[PSCustomObject]@{Archive=$archivePath; Files=$manifest.Count; AllEntryHashesVerified=$true; StagingRemoved=$stagingRemoved; Bytes=(Get-Item -LiteralPath $archivePath).Length; SHA256=(Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash} | Format-List

