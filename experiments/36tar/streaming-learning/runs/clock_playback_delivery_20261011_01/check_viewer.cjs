// Check generated snapshot scripts and user-visible timing/drop labels.
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const root = 'E:/VIT Model/runs';
const reports = [];
for (const [name, dropped, received] of [
  ['clock_normal_gpu_20261011_01', 0, 12],
  ['clock_overload_gpu_20261011_01', 4, 8],
]) {
  const html = fs.readFileSync(path.join(root, name, 'viewer.html'), 'utf8');
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x => x[1]);
  const elements = {};
  const context = {
    window: {},
    document: {
      getElementById(id) { return elements[id] ||= { children: [], replaceChildren() { this.children = []; }, append(x) { this.children.push(x); } }; },
      createElement() { return {}; },
    },
  };
  for (const script of scripts) vm.runInNewContext(script, context, { timeout: 1000 });
  if (!elements.status.textContent.includes(`丢帧：${dropped}`)) throw Error('Drop label mismatch');
  if (!elements.status.textContent.includes(`模型已收到：${received} 帧`)) throw Error('Received label mismatch');
  if (!elements.frame.src.startsWith('data:image/jpeg;base64,')) throw Error('Snapshot frame missing');
  if (!elements.answers.children.every(x => x.textContent.includes('回答滞后'))) throw Error('Answer timing label missing');
  reports.push({ name, scripts: scripts.length, status: elements.status.textContent, answerCount: elements.answers.children.length, passed: true });
}
fs.writeFileSync(path.join(root, 'clock_playback_delivery_20261011_01/viewer_check.json'), JSON.stringify(reports, null, 2));
console.log(JSON.stringify(reports));
