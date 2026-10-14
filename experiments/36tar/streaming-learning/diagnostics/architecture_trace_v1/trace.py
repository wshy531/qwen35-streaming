"""Six synthetic frames through the real processor and a tiny random CPU model."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACTIVE = HERE.parents[1] / 'code/qwen35-streaming-s4'
sys.path.insert(0, str(ACTIVE))
import numpy as np
import torch
from transformers import AutoProcessor
from streaming.demo import FIELDS
from streaming.online import OnlinePairSession
from streaming.window import FramePacket
from tests.test_online import small_model


def fingerprint(session):
    fields = {}
    for name in FIELDS:
        tensors = getattr(session.main.cache, name)
        digest = hashlib.sha256()
        shapes = []
        for tensor in tensors:
            if tensor is None:
                digest.update(b'None'); shapes.append(None)
            else:
                value = tensor.detach().cpu().contiguous()
                digest.update(str((tuple(value.shape), value.dtype)).encode())
                digest.update(value.numpy().tobytes())
                shapes.append(list(value.shape))
        fields[name] = {'sha256': digest.hexdigest(), 'shapes': shapes,
                        'nonempty_layers': sum(t is not None for t in tensors)}
    return {'status': session.status(), 'fields': fields}


@torch.inference_mode()
def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    session = OnlinePairSession(model, processor, canvas=(32,32), max_tokens=256,
                                max_answer_tokens=2, resize_mode='letterbox', video_logits='none')
    rows = []
    def record(step, explanation, **extra):
        rows.append({'step': step, 'explanation': explanation, **fingerprint(session), **extra})
    try:
        record('fresh', '还没接收画面，缓存为空。')
        for i in range(4):
            session.append(FramePacket(i, float(i), np.full((32,32,3), 30+i*30, np.uint8), i+1))
            record('frame_'+str(i), '等待第二帧，无模型计算。' if i%2==0 else '一对新帧写入，位置和混合状态一起前进。')
            if i == 1:
                before = fingerprint(session)
                branch = session.main.fork()
                shared = [a.data_ptr() == b.data_ptr() for name in FIELDS
                          for a,b in zip(getattr(session.main.cache,name),getattr(branch.cache,name))
                          if a is not None]
                if not shared or any(shared): raise AssertionError('Fork must own independent mixed tensors')
                del branch
                answer = session.ask('Describe the visible subject.', 2)
                if fingerprint(session) != before: raise AssertionError('Question changed main state')
                record('question', '问答使用独立副本；主状态四类张量与计数逐位不变。随机回答不代表识别能力。',
                       main_unchanged=True, fork_storage_independent=True, generated_tokens=len(answer['generated_ids']))
        pre_reset = fingerprint(session)
        session.reset()
        empty = fingerprint(session)
        if session.main.seen_tokens or any(x['nonempty_layers'] for x in empty['fields'].values()):
            raise AssertionError('Reset left old state')
        record('reset', '清空全部缓存和位置，旧历史不再保留。', previous_tokens=pre_reset['status']['seen_tokens'])
        for i in range(4,6):
            session.append(FramePacket(i, float(i), np.full((32,32,3), 30+i*30, np.uint8), i+1))
            record('frame_'+str(i), '在新epoch中等待。' if i==4 else '从空状态建立新的历史；视频来源时间仍是4–5秒。')
        report = {'scope':'Actual CPU tiny random weights + official processor + six synthetic frames; no semantic ability claim',
                  'layers':model.config.text_config.num_hidden_layers, 'steps':rows, 'passed':True,
                  'source_hashes':{str(p.relative_to(ACTIVE)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(ACTIVE.rglob('*.py'))}}
        (args.out/'trace.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        cells = []
        for row in rows:
            counts = ', '.join(name+':'+str(value['nonempty_layers']) for name,value in row['fields'].items())
            cells.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in
                          (row['step'],row['status']['seen_tokens'],row['status']['next_rope_position'],counts,row['explanation']))+'</tr>')
        page = '<!doctype html><meta charset="utf-8"><title>六帧状态轨迹</title><style>body{font:17px system-ui;max-width:1200px;margin:30px auto}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>六帧状态轨迹 · CPU机制演示</h1><p>真实运行的微型随机模型；不证明真实权重识别能力。缓存哈希与形状详见trace.json。</p><table><tr><th>步骤</th><th>累计Token</th><th>下一位置</th><th>非空状态层</th><th>解释</th></tr>'+''.join(cells)+'</table>'
        (args.out/'trace.html').write_text(page,encoding='utf-8')
        print(json.dumps({'passed':True,'steps':len(rows),'layers':report['layers'],'out':str(args.out)},ensure_ascii=False))
    finally:
        session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
