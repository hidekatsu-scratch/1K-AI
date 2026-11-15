var CFG = {
  timeLimitMs: 15000,
  evolveTimeLimitMs: 45000,
  maxNodes: 300,
  maxConstants: 100,
  constMax: 1000,
  stopToken: 2,
  eosToken: 1,
  trainRatio: 0.8,
  seed: 2026,
  autoDownload: false,
  progressEvery: 10000,
  gaPopSize: 24,
  gaEliteCount: 4,
  gaCrossoverRate: 0.6,
  corpus: "A fox one day spied a beautiful bunch of ripe grapes hanging from a vine. The grapes were ripe and juicy. The fox wanted them for his dinner. He jumped up but could not reach the grapes. He tried again and again. At last he gave up. He said the grapes are probably sour anyway. It is easy to despise what you cannot get. A crow was sitting on a branch. He had a piece of cheese in his beak. A fox saw the crow. He wanted the cheese. He said to the crow. What a beautiful voice you have. Please sing for me. The crow opened his beak to sing. The cheese fell down. The fox caught the cheese and ran away. The crow was very sad. He learned a lesson. Do not trust flatterers. A hare was very proud of his speed. He laughed at the tortoise for being slow. The tortoise said let us have a race. The hare laughed but agreed. The race began. The hare ran very fast. He was far ahead of the tortoise. He decided to take a nap. The tortoise kept walking slowly. He passed the sleeping hare. The tortoise reached the finish line. The hare woke up and ran but it was too late. Slow and steady wins the race."
};

var _dagSeed = CFG.seed >>> 0;
var _opSeed = CFG.seed >>> 0;

function randDag(){ _dagSeed = (Math.imul(_dagSeed,1664525)+1013904223)>>>0; return _dagSeed/4294967296; }
function randOp(){ _opSeed = (Math.imul(_opSeed,1664525)+1013904223)>>>0; return _opSeed/4294967296; }

function shuffle(a){
  for(var i=a.length-1;i>0;i--){
    var j=Math.floor(randDag()*(i+1));
    var t=a.at(i);
    a.splice(i,1,a.at(j));
    a.splice(j,1,t);
  }
  return a;
}

var UNARY_OPS = ['abs','round','floor','ceil','sqrt','sin','cos','tan','asin','acos','atan','ln','log','exp','exp10','not','len'];
var BINARY_OPS = ['add','sub','mul','div','mod','gt','lt','eq','and','or','concat','extract','contains','rand'];

function clampNum(v){
  if(!isFinite(v)) return 0;
  if(v>CFG.constMax) return CFG.constMax;
  if(v<-CFG.constMax) return -CFG.constMax;
  return v;
}

function toStr(v){
  if(Number.isInteger(v)) return String(v);
  var s = String(v);
  return s.length>30 ? s.slice(0,30) : s;
}

function applyUnary(op,a){
  var t,v;
  switch(op){
    case 'abs': return Math.abs(a);
    case 'round': return Math.round(a);
    case 'floor': return Math.floor(a);
    case 'ceil': return Math.ceil(a);
    case 'sqrt': return a<0?0:Math.sqrt(a);
    case 'sin': return Math.sin(a);
    case 'cos': return Math.cos(a);
    case 'tan': t=Math.tan(a); return isFinite(t)?t:0;
    case 'asin': v=Math.asin(a); return isNaN(v)?0:v;
    case 'acos': v=Math.acos(a); return isNaN(v)?0:v;
    case 'atan': return Math.atan(a);
    case 'ln': return a<=0?0:Math.log(a);
    case 'log': return a<=0?0:Math.log10(a);
    case 'exp': v=Math.exp(a); return isFinite(v)?v:0;
    case 'exp10': v=Math.pow(10,a); return isFinite(v)?v:0;
    case 'not': return a!==0?0:1;
    case 'len': return toStr(a).length;
  }
  return 0;
}

function applyBinary(op,a,b){
  var v,s,n,ch,d,lo,hi;
  switch(op){
    case 'add': return a+b;
    case 'sub': return a-b;
    case 'mul': return a*b;
    case 'div': return b===0?0:a/b;
    case 'mod': return b===0?0:a%b;
    case 'gt': return a>b?1:0;
    case 'lt': return a<b?1:0;
    case 'eq': return a===b?1:0;
    case 'and': return (a!==0&&b!==0)?1:0;
    case 'or': return (a!==0||b!==0)?1:0;
    case 'concat': v=parseFloat(toStr(a)+toStr(b)); return isFinite(v)?v:0;
    case 'extract':
      s=toStr(a); n=Math.floor(b);
      if(n<1||n>s.length) return 0;
      ch=s.charAt(n-1); d=parseInt(ch,10);
      return isNaN(d)?ch.charCodeAt(0):d;
    case 'contains': return toStr(a).indexOf(toStr(b))>=0?1:0;
    case 'rand': lo=Math.min(a,b); hi=Math.max(a,b); return lo+randOp()*(hi-lo);
  }
  return 0;
}

function buildVocab(text){
  var freq = {};
  for(var i=0; i<text.length; i++){
    var ch = text.charAt(i);
    freq[ch] = (freq[ch] || 0) + 1;
  }
  var chars = Object.keys(freq).sort(function(a,b){ return freq[b] - freq[a]; });
  var charToId = {};
  for(var id=0; id<chars.length; id++){
    charToId[chars.at(id)] = id;
  }
  return { chars: chars, charToId: charToId, size: chars.length };
}

var VOCAB = buildVocab(CFG.corpus);

function randomDAG(){
  var nodes = [{op:'input',idx:0},{op:'input',idx:1},{op:'input',idx:2},{op:'input',idx:3}];
  var total = 6 + Math.floor(randDag()*Math.max(1, CFG.maxNodes-6));
  var consts = 0, i, r, op, idx1, idx2, outStart, outputs, k;
  for(i=4;i<total;i++){
    r = randDag();
    if(r<0.15 && consts<CFG.maxConstants){
      nodes.push({op:'const',val:(randDag()*2-1)*CFG.constMax});
      consts++;
    } else if(r<0.45){
      op = UNARY_OPS.at(Math.floor(randDag()*UNARY_OPS.length));
      idx1 = Math.floor(randDag()*i);
      nodes.push({op:op,args:[idx1]});
    } else {
      op = BINARY_OPS.at(Math.floor(randDag()*BINARY_OPS.length));
      idx1 = Math.floor(randDag()*i);
      idx2 = Math.floor(randDag()*i);
      nodes.push({op:op,args:[idx1,idx2]});
    }
  }
  outStart = Math.max(4, Math.floor(total*0.7));
  outputs = [];
  for(k=0;k<4;k++){
    if(randDag()<0.7) outputs.push(outStart+Math.floor(randDag()*(total-outStart)));
    else outputs.push(4+Math.floor(randDag()*(total-4)));
  }
  var embed = [];
  for(var v=0; v<VOCAB.size; v++){
    embed.push((randDag()*2 - 1) * 10.0);
  }
  return {nodes:nodes, outputs:outputs, embed:embed};
}

function prune(dag){
  var nodes = dag.nodes, outputs = dag.outputs;
  var needed = {};
  var stack = outputs.slice();
  var idx, nd, i, a;
  while(stack.length){
    idx = stack.pop();
    if(idx<4 || needed[idx]) continue;
    needed[idx] = true;
    nd = nodes.at(idx);
    if(nd && nd.args){
      for(i=0;i<nd.args.length;i++){
        a = nd.args.at(i);
        if(a>=4 && !needed[a]) stack.push(a);
      }
    }
  }
  return Object.keys(needed).map(Number).sort(function(a,b){return a-b;});
}

function evaluateStep(dag,pruned,vals,x){
  var nodes = dag.nodes, outputs = dag.outputs;
  vals.set(x, 0);
  var k, i, nd, v, a0, a1;
  for(k=0;k<pruned.length;k++){
    i = pruned.at(k); nd = nodes.at(i);
    if(!nd) continue;
    if(nd.op==='const') v = nd.val;
    else if(nd.args.length===1){ a0 = nd.args.at(0); v = applyUnary(nd.op, vals.at(a0)); }
    else { a0 = nd.args.at(0); a1 = nd.args.at(1); v = applyBinary(nd.op, vals.at(a0), vals.at(a1)); }
    vals.set([clampNum(v)], i);
  }
  return [vals.at(outputs.at(0)), vals.at(outputs.at(1)), vals.at(outputs.at(2)), vals.at(outputs.at(3))];
}

function nearestTokenId(y, embed){
  if(!isFinite(y)) return 0;
  var bestId = 0, bestDist = Infinity;
  for(var i=0; i<embed.length; i++){
    var dist = Math.abs(y - embed.at(i));
    if(dist < bestDist){
      bestDist = dist;
      bestId = i;
    }
  }
  return bestId;
}

function splitSentences(text){
  var res = [], cur = "", i, ch;
  for(i=0;i<text.length;i++){
    ch = text.charAt(i);
    cur += ch;
    if(ch==='.'||ch==='!'||ch==='?'){
      if(cur.replace(/\s+/g,'').length>1) res.push(cur.replace(/\s+$/,'')+' ');
      cur = "";
    }
  }
  if(cur.replace(/\s+/g,'').length>1) res.push(cur);
  return res;
}

function fmt(s){
  return String(s).replace(/[^\x20-\x7E]/g, function(c){
    var h = c.charCodeAt(0).toString(16);
    if(h.length<2) h = '0'+h;
    return '\\x'+h;
  });
}

function evalEmbedScore(d, p, sents){
  _opSeed = CFG.seed >>> 0;
  var vals = new Float64Array(d.nodes.length);
  var totalScore = 0, maxScore = 0, uniqueIds = new Set();
  var penalty = 0;
  var c2i = VOCAB.charToId;
  var emb = d.embed;

  for(var si=0; si<sents.length; si++){
    var sent = sents.at(si);
    if(sent.length < 4) continue;

    var init = evaluateStep(d, p, vals, [-99, 0, 0, 0]);
    var s1=init.at(1), s2=init.at(2), s3=init.at(3);
    var streak1 = 0;
    for(var t=0; t<sent.length-1; t++){
      var inId = c2i[sent.charAt(t)] ?? 0;
      var targetId = c2i[sent.charAt(t+1)] ?? 0;
      var out = evaluateStep(d, p, vals, [emb.at(inId), s1, s2, s3]);
      var predId = nearestTokenId(out.at(0), emb);
      uniqueIds.add(predId);
      if(predId === targetId){
        streak1++;
        totalScore += 1.0 * (streak1 * streak1);
      } else {
        streak1 = 0;
      }
      maxScore += 1.0;
      s1=out.at(1); s2=out.at(2); s3=out.at(3);
    }

    var init2 = evaluateStep(d, p, vals, [-99, 0, 0, 0]);
    var a1=init2.at(1), a2=init2.at(2), a3=init2.at(3);
    var firstChar = sent.charAt(0);
    var feedVal = emb.at(c2i[firstChar] ?? 0);
    var streak2 = 0, prevId = -1, repeatCnt = 0, selfUnique = new Set();
    var genText = firstChar;

    for(var k=0; k<sent.length-1; k++){
      var out2 = evaluateStep(d, p, vals, [feedVal, a1, a2, a3]);
      var predId2 = nearestTokenId(out2.at(0), emb);
      if(predId2 === prevId){
        repeatCnt++;
        if(repeatCnt >= 2) penalty += 5.0;
      } else {
        repeatCnt = 0;
      }
      prevId = predId2;
      selfUnique.add(predId2);
      uniqueIds.add(predId2);

      var predChar = VOCAB.chars.at(predId2);
      genText += predChar;

      var parts = genText.split(' ');
      var wLenCur = parts.at(parts.length - 1).length;
      if(wLenCur > 12) penalty += (wLenCur - 12) * 2.0;

      if(predChar === ' '){
        var words = genText.trim().split(/\s+/);
        var wLen = words.length;
        if(wLen >= 3){
          var w1 = words.at(wLen - 1);
          var w2 = words.at(wLen - 2);
          var w3 = words.at(wLen - 3);
          if(w1 === w2 && w2 === w3) penalty += 15.0;
        }
      }

      var targetId2 = c2i[sent.charAt(k+1)] ?? 0;
      if(predId2 === targetId2){
        streak2++;
        totalScore += 2.5 * (streak2 * streak2);
      } else {
        streak2 = 0;
      }
      maxScore += 2.5;
a1=out2.at(1); a2=out2.at(2); a3=out2.at(3);
if(k < 1){
var nextCorrect = sent.charAt(k+1);
feedVal = emb.at(c2i[nextCorrect] ?? 0);
genText = sent.slice(0, k+2);
} else {
feedVal = emb.at(predId2);
}
}
if(sent.length > 10 && selfUnique.size < 4) penalty += 20.0;
}
if(uniqueIds.size < 6) return -1000;
return ((totalScore - penalty) / maxScore) * 100;
}
function mutateDAG(parent){
var nodes = parent.nodes.map(function(nd){
if(nd.op === 'input') return {op:'input', idx:nd.idx};
if(nd.op === 'const') return {op:'const', val:nd.val};
return {op:nd.op, args:nd.args.slice()};
});
var outputs = parent.outputs.slice();
var embed = parent.embed ? parent.embed.slice() : VOCAB.chars.map(function(_, idx){ return idx * 0.5; });
var pruned = prune(parent);
var mutations = 2 + Math.floor(randDag() * 4);
for(var m=0; m<mutations; m++){
var r = randDag();
if(r < 0.40){
var vIdx = Math.floor(randDag() * embed.length);
var re = randDag();
if(re < 0.65){
embed.splice(vIdx, 1, clampNum(embed.at(vIdx) + (randDag()*2 - 1) * 0.5));
} else if(re < 0.90){
embed.splice(vIdx, 1, clampNum(embed.at(vIdx) + (randDag()*2 - 1) * 3.0));
} else {
var vIdx2 = Math.floor(randDag() * embed.length);
var tmp = embed.at(vIdx);
embed.splice(vIdx, 1, embed.at(vIdx2));
embed.splice(vIdx2, 1, tmp);
}
} else if(r < 0.55 || pruned.length === 0){
var outIdx = Math.floor(randDag() * 4);
var newTarget = 4 + Math.floor(randDag() * (nodes.length - 4));
outputs.splice(outIdx, 1, newTarget);
} else {
var targetIdx = (randDag() < 0.7)
? pruned.at(Math.floor(randDag() * pruned.length))
: (4 + Math.floor(randDag() * (nodes.length - 4)));
var nd = nodes.at(targetIdx);
if(!nd || nd.op === 'input') continue;
if(nd.op === 'const'){
if(randDag() < 0.6) nd.val = clampNum(nd.val + (randDag()*2 - 1) * 5.0);
else nd.val = (randDag()*2 - 1) * CFG.constMax;
} else if(nd.args.length === 1){
if(randDag() < 0.5) nd.op = UNARY_OPS.at(Math.floor(randDag() * UNARY_OPS.length));
else nd.args.splice(0, 1, Math.floor(randDag() * targetIdx));
} else {
if(randDag() < 0.5) nd.op = BINARY_OPS.at(Math.floor(randDag() * BINARY_OPS.length));
else {
var argPos = Math.floor(randDag() * 2);
nd.args.splice(argPos, 1, Math.floor(randDag() * targetIdx));
}
}
}
}
return {nodes:nodes, outputs:outputs, embed:embed};
}
function crossoverDAG(parentA, parentB){
var lenA = parentA.nodes.length;
var lenB = parentB.nodes.length;
var minLen = Math.min(lenA, lenB);
var cut = 4 + Math.floor(randDag() * Math.max(1, minLen - 4));
var nodes = [];
for(var i=0; i<lenA; i++){
var src = (i < cut || i >= lenB) ? parentA.nodes.at(i) : parentB.nodes.at(i);
if(src.op === 'input') nodes.push({op:'input', idx:src.idx});
else if(src.op === 'const') nodes.push({op:'const', val:src.val});
else {
var safeArgs = src.args.map(function(a){ return a >= i ? (a % i) : a; });
nodes.push({op:src.op, args:safeArgs});
}
}
var outputs = [];
for(var k=0; k<4; k++){
var pick = (randDag() < 0.5) ? parentA.outputs.at(k) : parentB.outputs.at(k);
outputs.push(pick >= nodes.length ? (4 + (pick % Math.max(1, nodes.length - 4))) : pick);
}
var embed = [];
var embA = parentA.embed || [];
var embB = parentB.embed || embA;
for(var v=0; v<VOCAB.size; v++){
embed.push((randDag() < 0.5 ? embA.at(v) : embB.at(v)) ?? 0);
}
return {nodes:nodes, outputs:outputs, embed:embed};
}
function generateFrom(dag, pruned, prompt, maxLen){
prompt = prompt || 'The ';
maxLen = maxLen || 80;
if(!dag || !dag.embed) return '';
_opSeed = CFG.seed >>> 0;
var vals = new Float64Array(dag.nodes.length);
var p = pruned || prune(dag);
var c2i = VOCAB.charToId;
var emb = dag.embed;
var init = evaluateStep(dag, p, vals, [-99, 0, 0, 0]);
var s1=init.at(1), s2=init.at(2), s3=init.at(3), nextY=init.at(0);
for(var t=0; t<prompt.length; t++){
var inId = c2i[prompt.charAt(t)] ?? 0;
var out = evaluateStep(dag, p, vals, [emb.at(inId), s1, s2, s3]);
s1=out.at(1); s2=out.at(2); s3=out.at(3); nextY=out.at(0);
}
var res = "";
for(var g=0; g<maxLen; g++){
var predId = nearestTokenId(nextY, emb);
res += VOCAB.chars.at(predId);
var out2 = evaluateStep(dag, p, vals, [emb.at(predId), s1, s2, s3]);
s1=out2.at(1); s2=out2.at(2); s3=out2.at(3); nextY=out2.at(0);
}
return res;
}
var _currentDag = null;
var _currentPruned = null;
function saveBest(dag,trainScore,valScore,iter){
var obj = { timestamp: Date.now(), iter: iter, trainScore: trainScore, valScore: valScore, dag: dag };
try { localStorage.setItem('randomDagLM_best', JSON.stringify(obj)); }
catch(e){ console.warn('[save] failed:', e); }
}
function loadModel(){
try {
var data = localStorage.getItem('randomDagLM_best');
if(!data){ console.warn('[load] no model saved'); return null; }
var obj = JSON.parse(data);
_currentDag = obj.dag;
_currentPruned = prune(_currentDag);
console.log('[load] iter='+obj.iter+' train='+(obj.trainScore||0).toFixed(2)+' val='+(obj.valScore||0).toFixed(2));
return _currentDag;
} catch(e){ console.error('[load] failed:', e); return null; }
}
function downloadModel(){
try {
var data = localStorage.getItem('randomDagLM_best');
if(!data){ console.warn('[download] no model'); return; }
var blob = new Blob([data], {type:'application/json'});
var url = URL.createObjectURL(blob);
var a = document.createElement('a');
a.href = url;
a.download = 'randomDagLM_' + Date.now() + '.json';
document.body.appendChild(a);
a.click();
document.body.removeChild(a);
setTimeout(function(){ URL.revokeObjectURL(url); }, 2000);
console.log('[download] started');
} catch(e){ console.error('[download] failed:', e); }
}
function generate(prompt,maxLen){
if(!_currentDag) loadModel();
return generateFrom(_currentDag, _currentPruned, prompt, maxLen);
}
async function evolveOnly(durationMs){
durationMs = durationMs || CFG.evolveTimeLimitMs || 45000;
if(!_currentDag) loadModel();
if(!_currentDag){ console.warn('[evolve] no base model'); return null; }
_dagSeed = CFG.seed >>> 0;
var sentences = splitSentences(CFG.corpus);
shuffle(sentences);
var trainCount = Math.max(1, Math.min(sentences.length-1, Math.floor(sentences.length*CFG.trainRatio)));
var trainSet = sentences.slice(0, trainCount);
var valSet = sentences.slice(trainCount);
var currDag = _currentDag;
var currTrain = evalEmbedScore(currDag, prune(currDag), trainSet);
var bestDag = currDag;
var bestTrain = currTrain;
var bestVal = valSet.length>0 ? evalEmbedScore(bestDag, prune(bestDag), valSet) : bestTrain;
console.log('[evolve] start! baseScore='+bestTrain.toFixed(2));
var t0 = Date.now(), lastYield = t0, gen = 0, wins = 0;
while(Date.now()-t0 < durationMs){
var progress = (Date.now() - t0) / durationMs;
var temp = 2.0 * (1.0 - progress);
var child = mutateDAG(currDag);
var p = prune(child);
if(p.length === 0) continue;
var sc = evalEmbedScore(child, p, trainSet);
gen++;
if(sc <= -999) continue;
var diff = sc - currTrain;
if(diff > 0 || (temp > 0.05 && randDag() < Math.exp(diff / temp))){
currDag = child;
currTrain = sc;
if(sc > bestTrain){
var vSc = valSet.length>0 ? evalEmbedScore(child, p, valSet) : sc;
bestTrain = sc;
bestVal = vSc;
bestDag = child;
_currentDag = child;
_currentPruned = p;
wins++;
saveBest(child, sc, vSc, gen);
console.log('[evolve] gen='+gen+' (wins='+wins+') NEW BEST score='+sc.toFixed(2)+' val='+vSc.toFixed(2));
}
}
if(gen % CFG.progressEvery === 0){
console.log('[evolve] gen='+gen+' wins='+wins+' curr='+currTrain.toFixed(2)+' best='+bestTrain.toFixed(2));
}
if(Date.now()-lastYield > 100){
await new Promise(function(r){ setTimeout(r,0); });
lastYield = Date.now();
}
}
console.log('[evolve] done. gens='+gen+' wins='+wins+' bestScore='+bestTrain.toFixed(2));
console.log('[evolve] sample="'+fmt(generateFrom(bestDag, prune(bestDag), 'The ', 60))+'"');
return bestDag;
}
async function search(){
VOCAB = buildVocab(CFG.corpus);
_dagSeed = CFG.seed >>> 0;
var sentences = splitSentences(CFG.corpus);
shuffle(sentences);
var trainCount = Math.max(1, Math.min(sentences.length-1, Math.floor(sentences.length*CFG.trainRatio)));
var trainSet = sentences.slice(0, trainCount);
var valSet = sentences.slice(trainCount);
console.log('[search] Phase 1: Random Search ('+CFG.timeLimitMs+'ms)');
var t0 = Date.now(), lastYield = t0;
var bestTrain = -999, bestVal = -999, bestDag = null, iter = 0;
while(Date.now()-t0 < CFG.timeLimitMs){
var dag = randomDAG();
var pruned = prune(dag);
if(pruned.length===0) continue;
var sc = evalEmbedScore(dag, pruned, trainSet);
iter++;
if(sc > bestTrain){
var vSc = valSet.length>0 ? evalEmbedScore(dag, pruned, valSet) : sc;
bestTrain = sc;
bestVal = vSc;
bestDag = dag;
_currentDag = dag;
_currentPruned = pruned;
saveBest(dag, sc, vSc, iter);
console.log('[search] iter='+iter+' NEW BEST score='+sc.toFixed(2)+' val='+vSc.toFixed(2));
}
if(iter % CFG.progressEvery === 0){
console.log('[search] iter='+iter+' bestScore='+bestTrain.toFixed(2));
}
if(Date.now()-lastYield > 100){
await new Promise(function(r){ setTimeout(r,0); });
lastYield = Date.now();
}
}
console.log('[search] Phase 1 done. iters='+iter+' bestScore='+bestTrain.toFixed(2));
if(bestDag && CFG.evolveTimeLimitMs > 0){
await evolveOnly(CFG.evolveTimeLimitMs);
}
return _currentDag;
}
var AI = {
config: CFG,
vocab: function(){ return VOCAB; },
search: search,
evolve: evolveOnly,
generate: generate,
download: downloadModel,
load: loadModel,
embed: function(){
if(!_currentDag || !_currentDag.embed){ console.log('No embed model'); return; }
var list = VOCAB.chars.map(function(ch, i){
return { char: ch, val: Number(_currentDag.embed.at(i).toFixed(3)) };
}).sort(function(a,b){ return a.val - b.val; });
console.log('[Learned Embeddings]', list.map(function(x){ return JSON.stringify(x.char)+':'+x.val; }).join(', '));
}
};
console.log('[RandomDAG-LM v2] loaded. Commands: AI.search() / AI.evolve() / AI.generate("The ") / AI.embed() / AI.download()');
