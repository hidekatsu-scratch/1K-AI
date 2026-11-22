// test_fast.js
// 1K-AI 高速学習版 (Fast Learning & Adaptive Search)

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
  progressEvery: 2000, // 進捗ログの頻度を増加
  gaPopSize: 24,
  gaEliteCount: 4,
  gaCrossoverRate: 0.6,
  corpus: "A fox one day spied a beautiful bunch of ripe grapes hanging from a vine. The grapes were ripe and juicy. The fox wanted them for his dinner. He jumped up but could not reach the grapes. He tried again and again. At last he gave up. He said the grapes are probably sour anyway. It is easy to despise what you cannot get. A crow was sitting on a branch. He had a piece of cheese in his beak. A fox saw the crow. He wanted the cheese. He said to the crow. What a beautiful voice you have. Please sing for me. The crow opened his beak to sing. The cheese fell down. The fox caught the cheese and ran away. The crow was very sad. He learned a lesson. Do not trust flatterers. A hare was very proud of his speed. He laughed at the tortoise for being slow. The tortoise said let us have a race. The hare laughed but agreed. The race began. The hare ran very fast. He was far ahead of the tortoise. He decided to take a nap. The tortoise kept walking slowly. He passed the sleeping hare. The tortoise reached the finish line. The hare woke up and ran but it was too late. Slow and steady wins the race."
};

var _dagSeed = CFG.seed >>> 0;
var _opSeed = CFG.seed >>> 0;

function randDag(){ _dagSeed = (Math.imul(_dagSeed, 1664525) + 1013904223) >>> 0; return _dagSeed / 4294967296; }
function randOp(){ _opSeed = (Math.imul(_opSeed, 1664525) + 1013904223) >>> 0; return _opSeed / 4294967296; }

function shuffle(a){ for(var i = a.length - 1; i > 0; i--){ var j = Math.floor(randDag() * (i + 1)); var t = a.at(i); a.splice(i, 1, a.at(j)); a.splice(j, 1, t); } return a; }

var UNARY_OPS = ['abs','round','floor','ceil','sqrt','sin','cos','tan','asin','acos','atan','ln','log','exp','exp10','not','len'];
var BINARY_OPS = ['add','sub','mul','div','mod','gt','lt','eq','and','or','concat','extract','contains','rand'];

function clampNum(v){
  if(!isFinite(v)) return 0;
  if(v > CFG.constMax) return CFG.constMax;
  if(v < -CFG.constMax) return -CFG.constMax;
  return v;
}

function toStr(v){ if(Number.isInteger(v)) return String(v); var s = String(v); return s.length > 30 ? s.slice(0, 30) : s; }

function applyUnary(op, a){
  var t, v;
  switch(op){
    case 'abs': return Math.abs(a);
    case 'round': return Math.round(a);
    case 'floor': return Math.floor(a);
    case 'ceil': return Math.ceil(a);
    case 'sqrt': return a < 0 ? 0 : Math.sqrt(a);
    case 'sin': return Math.sin(a);
    case 'cos': return Math.cos(a);
    case 'tan': t = Math.tan(a); return isFinite(t) ? t : 0;
    case 'asin': v = Math.asin(a); return isNaN(v) ? 0 : v;
    case 'acos': v = Math.acos(a); return isNaN(v) ? 0 : v;
    case 'atan': return Math.atan(a);
    case 'ln': return a <= 0 ? 0 : Math.log(a);
    case 'log': return a <= 0 ? 0 : Math.log10(a);
    case 'exp': v = Math.exp(a); return isFinite(v) ? v : 0;
    case 'exp10': v = Math.pow(10, a); return isFinite(v) ? v : 0;
    case 'not': return a !== 0 ? 0 : 1;
    case 'len': return toStr(a).length;
  }
  return 0;
}

function applyBinary(op, a, b){
  var v, s, n, ch, d, lo, hi;
  switch(op){
    case 'add': return a + b;
    case 'sub': return a - b;
    case 'mul': return a * b;
    case 'div': return b === 0 ? 0 : a / b;
    case 'mod': return b === 0 ? 0 : a % b;
    case 'gt': return a > b ? 1 : 0;
    case 'lt': return a < b ? 1 : 0;
    case 'eq': return a === b ? 1 : 0;
    case 'and': return (a !== 0 && b !== 0) ? 1 : 0;
    case 'or': return (a !== 0 || b !== 0) ? 1 : 0;
    case 'concat': return toStr(a) + toStr(b);
    case 'extract': s = toStr(a); n = Math.floor(b); return (n >= 0 && n < s.length) ? s.charCodeAt(n) : 0;
    case 'charAt': s = toStr(a); n = Math.floor(b); return (n >= 0 && n < s.length) ? s.charAt(n) : "";
    case 'indexOf': return toStr(a).indexOf(toStr(b));
    case 'slice': s = toStr(a); n = Math.floor(b); return (n >= 0 && n < s.length) ? s.slice(n) : "";
    case 'repeat': s = toStr(a); n = Math.floor(b); return n > 0 && n < 20 ? s.repeat(n) : s;
    case 'replace': return toStr(a).replace(toStr(b), "");
    case 'split': return toStr(a).split(toStr(b)).length;
    case 'len': return toStr(a).length;
    case 'charCode': return toStr(a).charCodeAt(0) || 0;
    case 'fromCharCode': return String.fromCharCode(a);
  }
  return 0;
}

function buildVocab(text){
  var freq = {};
  for(var i = 0; i < text.length; i++){
    var c = text.charAt(i);
    freq[c] = (freq[c] || 0) + 1;
  }
  var chars = Object.keys(freq).sort(function(a, b){ return freq[b] - freq[a]; });
  var charToId = {};
  for(var i = 0; i < chars.length; i++) charToId[chars[i]] = i;
  return { chars: chars, charToId: charToId, size: chars.length };
}

var VOCAB = null;

function randomDAG(){
  var n = 4 + Math.floor(randDag() * CFG.maxNodes);
  var nodes = [];
  for(var i = 0; i < 4; i++) nodes.push({ op: 'input', idx: i });
  for(var i = 4; i < n; i++){
    if(randDag() < 0.15){
      nodes.push({ op: 'const', val: clampNum((randDag() - 0.5) * 200) });
    } else {
      var isUnary = randDag() < 0.5;
      if(isUnary){
        var uop = UNARY_OPS[Math.floor(randDag() * UNARY_OPS.length)];
        nodes.push({ op: uop, args: [Math.floor(randDag() * i)] });
      } else {
        var bop = BINARY_OPS[Math.floor(randDag() * BINARY_OPS.length)];
        var a1 = Math.floor(randDag() * i);
        var a2 = Math.floor(randDag() * i);
        nodes.push({ op: bop, args: [a1, a2] });
      }
    }
  }
  var outputs = [];
  for(var k = 0; k < 4; k++) outputs.push(n - 4 + k);
  var embed = VOCAB.chars.map(function(_, idx){ return idx * 0.5; });
  return { nodes: nodes, outputs: outputs, embed: embed };
}

function prune(dag){
  var n = dag.nodes.length;
  var needed = new Set(dag.outputs);
  var changed = true;
  while(changed){
    changed = false;
    for(var i = n - 1; i >= 0; i--){
      if(needed.has(i)){
        var nd = dag.nodes[i];
        if(nd.args){
          for(var j = 0; j < nd.args.length; j++){
            var a = nd.args[j];
            if(a >= 4 && !needed.has(a)){ needed.add(a); changed = true; }
          }
        }
      }
    }
  }
  return Array.from(needed).sort(function(a, b){ return a - b; });
}

function evaluateStep(dag, pruned, vals, x){
  var nodes = dag.nodes, outputs = dag.outputs;
  vals.set(x, 0);
  var k, i, nd, v, a0, a1;
  for(k = 0; k < pruned.length; k++){
    i = pruned[k]; nd = nodes[i];
    switch(nd.op){
      case 'input': v = vals[nd.idx]; break;
      case 'const': v = nd.val; break;
      default:
        a0 = vals[nd.args[0]];
        if(nd.args.length === 1) v = applyUnary(nd.op, a0);
        else {
          a1 = vals[nd.args[1]];
          v = applyBinary(nd.op, a0, a1);
        }
        break;
    }
    vals[i] = clampNum(v);
  }
  return outputs.map(function(idx){ return vals[idx]; });
}

function splitSentences(text){
  var res = [];
  var cur = "";
  for(var i = 0; i < text.length; i++){
    var ch = text.charAt(i);
    cur += ch;
    if(ch === '.' || ch === '!' || ch === '?'){
      if(cur.replace(/\s+/g, '').length > 1) res.push(cur);
      cur = "";
    }
  }
  if(cur.replace(/\s+/g, '').length > 1) res.push(cur);
  return res;
}

function fmt(s){ return String(s).replace(/[^\x20-\x7E]/g, function(c){ var h = c.charCodeAt(0).toString(16); if(h.length < 2) h = '0' + h; return '\\x' + h; }); }

// 高速スコアリング（文字列割り当てを極力排除）
function evalFast(dag, p, sentences){
  _opSeed = CFG.seed >>> 0;
  var vals = new Float64Array(dag.nodes.length);
  var totalScore = 0, maxScore = 0;
  var c2i = VOCAB.charToId;
  var emb = dag.embed;

  for(var si = 0; si < sentences.length; si++){
    var sent = sentences[si];
    var out = evaluateStep(dag, p, vals, [0, 0, 0, 0]);
    var predId = out[0] | 0;
    for(var k = 1; k < sent.length; k++){
      var targetId = c2i[sent.charAt(k)] ?? 0;
      if(predId === targetId) totalScore += 1.0;
      maxScore += 1.0;

      var feedVal = emb[predId] || 0;
      out = evaluateStep(dag, p, vals, [feedVal, out[1], out[2], out[3]]);
      predId = out[0] | 0;
    }
  }
  return (totalScore / maxScore) * 100;
}

function mutateDAG(parent){
  var nodes = parent.nodes.map(function(nd){
    if(nd.op === 'input') return { op: 'input', idx: nd.idx };
    if(nd.op === 'const') return { op: 'const', val: nd.val };
    return { op: nd.op, args: nd.args.slice() };
  });
  var outputs = parent.outputs.slice();
  var embed = parent.embed ? parent.embed.slice() : VOCAB.chars.map(function(_, idx){ return idx * 0.5; });

  var pruned = prune(parent);
  var mutations = 2 + Math.floor(randDag() * 4);

  for(var m = 0; m < mutations; m++){
    var r = randDag();
    if(r < 0.4 && pruned.length > 10){
      var idx = pruned[Math.floor(randDag() * pruned.length)];
      if(randDag() < 0.5 && nodes[idx].op !== 'input' && nodes[idx].op !== 'const'){
        nodes[idx].op = randDag() < 0.5 ? UNARY_OPS[Math.floor(randDag() * UNARY_OPS.length)] : BINARY_OPS[Math.floor(randDag() * BINARY_OPS.length)];
      } else {
        nodes[idx].op = 'const';
        nodes[idx].val = clampNum((randDag() - 0.5) * 200);
        delete nodes[idx].args;
      }
    } else if(r < 0.7 && pruned.length > 10){
      var idx = pruned[Math.floor(randDag() * pruned.length)];
      if(nodes[idx].args){
        var argIdx = Math.floor(randDag() * nodes[idx].args.length);
        nodes[idx].args[argIdx] = Math.floor(randDag() * nodes.length);
      }
    } else if(r < 0.85 && outputs.length > 0){
      var outIdx = Math.floor(randDag() * outputs.length);
      outputs[outIdx] = Math.floor(randDag() * nodes.length);
    } else {
      var embIdx = Math.floor(randDag() * embed.length);
      embed[embIdx] = clampNum(embed[embIdx] + (randDag() - 0.5) * 5);
    }
  }

  return { nodes: nodes, outputs: outputs, embed: embed };
}

function generateFrom(dag, pruned, prefix, length){
  var vals = new Float64Array(dag.nodes.length);
  var c2i = VOCAB.charToId;
  var out = evaluateStep(dag, pruned, vals, [0, 0, 0, 0]);
  var cur = prefix;
  var last = prefix.slice(-1);
  var feed = last ? (c2i[last] || 0) * 0.5 : 0;
  for(var k = 0; k < length; k++){
    out = evaluateStep(dag, pruned, vals, [feed, out[1], out[2], out[3]]);
    var predId = out[0] | 0;
    var ch = VOCAB.chars[predId];
    cur += ch;
    feed = (predId) * 0.5;
  }
  return cur;
}

// 高速進化計算
async function evolveFast(durationMs){
  var sentences = splitSentences(CFG.corpus);
  shuffle(sentences);
  var trainCount = Math.max(1, Math.min(sentences.length - 1, Math.floor(sentences.length * CFG.trainRatio)));
  var trainSet = sentences.slice(0, trainCount);
  var valSet = sentences.slice(trainCount);

  var currDag = _currentDag;
  var currTrain = evalFast(currDag, prune(currDag), trainSet);
  var bestDag = currDag;
  var bestTrain = currTrain;
  var bestVal = valSet.length > 0 ? evalFast(bestDag, prune(bestDag), valSet) : bestTrain;

  console.log('[evolve-fast] start! baseScore=' + bestTrain.toFixed(2));

  var t0 = Date.now(), lastYield = t0, gen = 0, wins = 0;

  while(Date.now() - t0 < durationMs){
    var progress = (Date.now() - t0) / durationMs;
    var temp = 2.0 * (1.0 - progress);
    var child = mutateDAG(currDag);
    var p = prune(child);
    if(p.length === 0) continue;

    var sc = evalFast(child, p, trainSet);
    gen++;

    if(sc > currTrain || (temp > 0.05 && randDag() < Math.exp((sc - currTrain) / temp))){
      currDag = child;
      currTrain = sc;

      if(sc > bestTrain){
        var vSc = valSet.length > 0 ? evalFast(child, p, valSet) : sc;
        bestTrain = sc;
        bestVal = vSc;
        bestDag = child;
        _currentDag = child;
        _currentPruned = p;
        wins++;
        saveBest(child, sc, vSc, gen);
        console.log('[evolve-fast] gen=' + gen + ' (wins=' + wins + ') NEW BEST score=' + sc.toFixed(2) + ' val=' + vSc.toFixed(2));
      }
    }

    if(gen % CFG.progressEvery === 0){
      console.log('[evolve-fast] gen=' + gen + ' wins=' + wins + ' curr=' + currTrain.toFixed(2) + ' best=' + bestTrain.toFixed(2));
    }

    if(Date.now() - lastYield > 100){
      await new Promise(function(r){ setTimeout(r, 0); });
      lastYield = Date.now();
    }
  }

  console.log('[evolve-fast] done. gens=' + gen + ' wins=' + wins + ' bestScore=' + bestTrain.toFixed(2));
  return bestDag;
}

async function searchFast(){
  VOCAB = buildVocab(CFG.corpus);
  _dagSeed = CFG.seed >>> 0;

  var sentences = splitSentences(CFG.corpus);
  shuffle(sentences);
  var trainCount = Math.max(1, Math.min(sentences.length - 1, Math.floor(sentences.length * CFG.trainRatio)));
  var trainSet = sentences.slice(0, trainCount);
  var valSet = sentences.slice(trainCount);

  console.log('[search-fast] Phase 1: Random Search (' + CFG.timeLimitMs + 'ms');
  var t0 = Date.now(), lastYield = t0;
  var bestTrain = -999, bestVal = -999, bestDag = null, iter = 0;

  while(Date.now() - t0 < CFG.timeLimitMs){
    var dag = randomDAG();
    var pruned = prune(dag);
    if(pruned.length === 0) continue;

    var sc = evalFast(dag, pruned, trainSet);
    iter++;

    if(sc > bestTrain){
      var vSc = valSet.length > 0 ? evalFast(dag, pruned, valSet) : sc;
      bestTrain = sc;
      bestVal = vSc;
      bestDag = dag;
      _currentDag = dag;
      _currentPruned = pruned;
      saveBest(dag, sc, vSc, iter);
      console.log('[search-fast] iter=' + iter + ' NEW BEST score=' + sc.toFixed(2) + ' val=' + vSc.toFixed(2));
    }

    if(iter % CFG.progressEvery === 0){
      console.log('[search-fast] iter=' + iter + ' bestScore=' + bestTrain.toFixed(2));
    }

    if(Date.now() - lastYield > 100){
      await new Promise(function(r){ setTimeout(r, 0); });
      lastYield = Date.now();
    }
  }

  console.log('[search-fast] Phase 1 done. iters=' + iter + ' bestScore=' + bestTrain.toFixed(2));

  if(bestDag && CFG.evolveTimeLimitMs > 0){
    await evolveFast(CFG.evolveTimeLimitMs);
  }

  return _currentDag;
}

var _currentDag = null;
var _currentPruned = null;

function saveBest(dag, trainScore, valScore, iter){
  if(CFG.autoDownload){
    downloadModel(dag, trainScore, valScore, iter);
  } else {
    var best = {
      timestamp: Date.now(),
      iter: iter,
      trainScore: trainScore,
      valScore: valScore,
      dag: dag
    };
    localStorage.setItem('randomDagLM_best', JSON.stringify(best));
  }
}

function generate(prefix, length){
  if(!_currentDag || !_currentDag.embed){
    console.log('No model loaded. Please run AI.search() first.');
    return;
  }
  var out = generateFrom(_currentDag, prune(_currentDag), prefix, length || 60);
  return fmt(out);
}

function downloadModel(dag, trainScore, valScore, iter){
  var best = {
    timestamp: Date.now(),
    iter: iter,
    trainScore: trainScore,
    valScore: valScore,
    dag: dag
  };
  var blob = new Blob([JSON.stringify(best)], { type: 'application/json' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'model_' + new Date().toISOString().replace(/[:.]/g, '-') + '.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function loadModel(jsonStr){
  var obj = JSON.parse(jsonStr);
  if(obj.dag){
    _currentDag = obj.dag;
    _currentPruned = prune(_currentDag);
    console.log('[load] loaded! iter=' + obj.iter + ' train=' + obj.trainScore.toFixed(2) + ' val=' + obj.valScore.toFixed(2));
    return _currentDag;
  } else {
    console.error('[load] invalid model format');
    return null;
  }
}

var AI = {
  config: CFG,
  vocab: function(){ return VOCAB; },
  search: searchFast,
  evolve: evolveFast,
  generate: generate,
  download: function(){ if(_currentDag) downloadModel(_currentDag, 0, 0, 0); else console.log('No model to download'); },
  load: loadModel,
  embed: function(){ if(!_currentDag || !_currentDag.embed){ console.log('No embed model'); return; } var list = VOCAB.chars.map(function(ch, i){ return { char: ch, val: Number(_currentDag.embed.at(i).toFixed(3)) }; }).sort(function(a,b){ return a.val - b.val; }); console.log('[Learned Embeddings]', list.map(function(x){ return JSON.stringify(x.char)+':'+x.val; }).join(', ')); }
};

console.log('[RandomDAG-LM v3 Fast] loaded. Commands: AI.search() / AI.evolve() / AI.generate("The ") / AI.embed() / AI.download()');
