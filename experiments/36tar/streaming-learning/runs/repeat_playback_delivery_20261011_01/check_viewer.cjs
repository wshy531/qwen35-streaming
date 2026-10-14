// Execute the actual saved HTML scripts with a small DOM; no visual claim.
const fs = require('fs'), vm = require('vm');
const root = 'E:/VIT Model/runs/';
const html = fs.readFileSync(root+'repeat_playback_gpu_20261011_01/viewer.html','utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x=>x[1]);
const elements = {};
const context = {window:{}, document:{
  getElementById(id){return elements[id] ||= {children:[],replaceChildren(){this.children=[];},append(x){this.children.push(x);}};},
  createElement(){return {};}
}};
for (const script of scripts) vm.runInNewContext(script,context,{timeout:1000});
const status = elements.status.textContent;
for (const label of ['明确循环第 5 次','来源时间 19.02 秒','会话时间 119.02 秒','当前记忆从：116.02 秒','重置：19 次','丢帧：0','模型已收到：112 帧','本次运行已结束']) {
  if (!status.includes(label)) throw Error('Missing or wrong label: '+label);
}
if (!elements.frame.src.startsWith('data:image/jpeg;base64,')) throw Error('No final frame');
if (elements.answers.children.length !== 8 || !elements.answers.children.every(x=>x.textContent.includes('回答滞后'))) throw Error('Bounded answer display wrong');
const report = {passed:true,scripts:scripts.length,status,answers:8,scope:'Actual snapshot JS with DOM simulation; not browser visual acceptance'};
fs.writeFileSync(root+'repeat_playback_delivery_20261011_01/viewer_check.json',JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
