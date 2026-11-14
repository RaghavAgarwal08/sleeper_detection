// Collects SAE latents and Jacobian lens readouts for a behavior_v1 sample from Neuronpedia (gpt-oss-20b).
// No GPU needed. Paste into the browser console on any https://www.neuronpedia.org page (same origin, so no API
// key is used), after setting window.S to the sample written by the Python helper below:
//
//   python - <<'EOF'      # from the repo root; prints the sample used on 2026-11-14 (seed 0)
//   ... see make_sample() in sae_jlens_analysis.py's git history, or results/probe_live_inert/neuronpedia/sample.json
//   EOF
//
// Then run collect() and, when SPROG.running and JPROG.running are both false, copy(JSON.stringify(exportAll())).
// Endpoints (found from the site's own network calls on 2026-11-14; not a documented public API, may change):
//   POST /api/search-all   {modelId, sourceSet: "resid-post-aa", text, selectedLayers: ["<L>-resid-post-aa"],
//                           sortIndexes: [token index], ignoreBos, densityThreshold: -1, numResults}
//                          SAE layers available for gpt-oss-20b: 3, 7, 11, 15, 19, 23. Text only, about 1,200 tokens max.
//   POST /api/lens/prompt  {modelId, inputTokenIds, type: ["JACOBIAN_LENS", "LOGIT_LENS"], topN, temperature,
//                           numCompletionTokens, prependBos} -> NDJSON stream, one line per position with the top
//                          tokens at all 24 layers. Rate limited (header x-limit-remaining, about 240 per window).

const BASE = 'https://raw.githubusercontent.com/pndh/sleeper_detection/main/interp/data/behavior_v1/';

async function loadData() {
  const need = new Set([...S.L.flat(), ...S.F.map(x => x[0])]);
  const txt = await (await fetch(BASE + 'requests.jsonl')).text();
  window.REQ = {};
  for (const line of txt.split('\n')) {
    if (!line) continue;
    const id = line.slice(8, 24);
    if (need.has(id)) { const r = JSON.parse(line); REQ[r.id] = {messages: r.request.messages, tools: r.request.tools, input_ids: r.input_ids, positions: r.positions, entry_spans: r.entry_spans}; }
  }
  const lab = await (await fetch(BASE + 'labels.jsonl')).text();
  window.LAB = {};
  for (const line of lab.split('\n')) {
    if (!line) continue;
    const l = JSON.parse(line);
    if (need.has(l.id)) LAB[l.id] = {case_id: l.case_id, template_id: l.template_id, framing: l.framing, domain: l.domain, note: l.note_text, note_idx: l.extra_note_entry_index, action: l.action};
  }
}

// Harmony rendering for the text-only SAE endpoint. Tool schemas are shortened to names to stay under the length cap.
function render(r) {
  const m = r.messages;
  const sys = m.find(x => x.role === 'system').content, user = m.find(x => x.role === 'user').content;
  const call = m.find(x => x.role === 'assistant' && x.tool_calls).tool_calls[0].function, tool = m.find(x => x.role === 'tool').content;
  const tools = r.tools.map(t => 'type ' + t.function.name + ' = (_: any) => any;').join('\n');
  return "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-10-06\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.\nCalls to these tools must go to the commentary channel: 'functions'.<|end|>"
    + '<|start|>developer<|message|># Instructions\n\n' + sys + '\n\n# Tools\n\n## functions\n\nnamespace functions {\n\n' + tools + '\n\n} // namespace functions<|end|>'
    + '<|start|>user<|message|>' + user + '<|end|>'
    + '<|start|>assistant<|channel|>commentary to=functions.' + call.name + ' <|constrain|>json<|message|>' + call.arguments + '<|call|>'
    + '<|start|>functions.' + call.name + ' to=assistant<|channel|>commentary<|message|>' + tool + '<|end|>' + '<|start|>assistant';
}

async function searchAll(text, layer, sort, n) {
  for (let a = 0; a < 5; a++) {
    const r = await fetch('/api/search-all', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({modelId: 'gpt-oss-20b', sourceSet: 'resid-post-aa', text, selectedLayers: [layer + '-resid-post-aa'], sortIndexes: sort, ignoreBos: true, densityThreshold: -1, numResults: n})});
    if (r.ok) return await r.json();
    await new Promise(s => setTimeout(s, 1500 * (a + 1)));  // the endpoint returns intermittent 500s; retry
  }
  throw new Error('search-all failed');
}

async function collectSAE() {
  window.SAE = {}; window.SERR = []; window.SPROG = {done: 0, running: true};
  for (const id of Object.keys(REQ)) {
    try {
      const t = render(REQ[id]);
      const toks = (await searchAll(t, 15, [], 1)).tokens;
      const mre = toks.lastIndexOf('<|end|>') - 1;  // last token of the memory result
      const rec = {n_tok: toks.length, mre, mre_tok: toks[mre], L: {}};
      for (const L of [15, 19]) {
        const j = await searchAll(t, L, [mre], 100);
        rec.L[L] = j.result.map(x => [Number(x.index), x.values[mre], x.values[toks.length - 1]]);
      }
      SAE[id] = rec;
    } catch (e) { SERR.push([id, String(e)]); }
    SPROG.done++;
  }
  SPROG.running = false;
}

async function collectJLens() {
  window.JL = {}; window.JERR = []; window.JPROG = {done: 0, running: true, remaining: null};
  for (const id of Object.keys(REQ)) {
    if (JPROG.remaining !== null && JPROG.remaining < 12) { JPROG.stopped = 'rate limit'; break; }
    try {
      const r = REQ[id];
      const note = r.entry_spans.find(e => e.entry_index === LAB[id].note_idx) || r.entry_spans[0];
      const pos = {mre: r.positions.memory_result_end, last: r.positions.last_prompt, note: note.occurrences[0][1] - 1, user_end: r.positions.user_end};
      let resp;
      for (let a = 0; a < 4; a++) {
        resp = await fetch('/api/lens/prompt', {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({modelId: 'gpt-oss-20b', inputTokenIds: r.input_ids, type: ['JACOBIAN_LENS', 'LOGIT_LENS'], topN: 8, temperature: 0, numCompletionTokens: 1, prependBos: false})});
        if (resp.ok || resp.status === 429) break;
        await new Promise(s => setTimeout(s, 3000 * (a + 1)));
      }
      JPROG.remaining = Number(resp.headers.get('x-limit-remaining'));
      if (!resp.ok) { JERR.push([id, resp.status]); if (resp.status === 429) break; continue; }
      const want = new Map(Object.entries(pos).map(([k, v]) => [v, k]));
      const rec = {pos};
      for (const l of (await resp.text()).split('\n')) {
        if (!l.startsWith('{"kind": "token"')) continue;
        const p = Number(l.slice(29, 40).split(',')[0]);
        if (!want.has(p)) continue;
        const o = JSON.parse(l);
        rec[want.get(p)] = {tok: o.token, J: o.results.find(x => x.type === 'JACOBIAN_LENS'), Lg: o.results.find(x => x.type === 'LOGIT_LENS')};
      }
      JL[id] = rec;
    } catch (e) { JERR.push([id, String(e)]); }
    JPROG.done++;
  }
  JPROG.running = false;
}

async function collect() { await loadData(); collectSAE(); collectJLens(); }

// Feature explanations for a list of [layer, index] pairs, via the feature endpoint.
async function explanations(list) {
  const out = {};
  for (const [layer, idx] of list) {
    const r = await fetch(`/api/feature/gpt-oss-20b/${layer}-resid-post-aa/${idx}`);
    if (!r.ok) { out[`${layer}-${idx}`] = null; continue; }
    const j = await r.json();
    out[`${layer}-${idx}`] = (j.explanations && j.explanations[0] && j.explanations[0].description) || null;
    out[`${layer}-${idx}_top_logits`] = (j.pos_str || []).slice(0, 5).join(' | ');
  }
  return out;
}

function exportAll() { return {S, SAE, JL, LAB, SERR, JERR}; }
