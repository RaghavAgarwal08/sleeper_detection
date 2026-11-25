# Where to take this next: research directions after the November runs

November 25, 2026. Builds on `results/LITERATURE_REVIEW.md`. For each direction: the question, why nobody has answered it yet (checked against the literature), the experiment, models and benchmarks, rough GPU cost on RunPod (H100 at $4.40 an hour), and what a result would mean. About $900 of credit remains.

## What we can build on

* **A monitor that predicts obedience before the agent acts.** Within wording, under a full prompt text baseline and pre-registered tests, it holds on corrected labels (+0.12 AUROC over text).
* **The probe's position is not on the causal path.** The model reads the note directly.
* **A difference of means direction suppresses obedience on new notes** (0.22 to 0.12) without costing the user's task. This is a replication of CounterSteer in the memory channel.
* **A reusable pipeline** for any open model: prompts, capture, probes, steering, patching, saved generations.

The field already has presence detectors, steering defenses (CounterSteer, ARGUS), "readable is not causal" papers, and memory poisoning benchmarks without internals (MemoryDojo). The directions below avoid those crowded spots.

## Ranked directions

### 1. Can an agent obey without the obey signal? Adaptive attacks on obedience monitors

**Question.** If an attacker rewrites the planted note to keep execution high while pushing the probe's score down, does it succeed? If so, what does it cost the attacker in execution?

**Why it is open.**
* Obfuscated Activations (Bailey et al., ICLR 2026) and RL-Obfuscation (Gupta and Jenner 2025) show probes can be evaded, but for jailbreak harmfulness. RL-Obfuscation finds mean pooled linear probes are evadable (evasion 0.59 to 0.82) while max pooled and attention probes stay robust (0.01 to 0.06).
* Nobody has tested monitors that **predict agent execution**, where the attacker must keep a behaviour (the planted tool call) rather than produce harmful text.
* Dong et al. (2026) test adaptive attacks on exposure probes only with search, and do not retrain.
* This is exactly the "probe robustness" question in the fellowship's stated topic.

**Experiment.** Three attacker strengths on behavior_v1 notes and AgentDojo, scored by execution rate (8 rollouts) and probe recall at 1% false alarms:
1. **Black box rewriting.** An LLM attacker iterates note wordings with feedback on probe score and execution, in the style of AutoDojo and AdvPrompter.
2. **White box hard text.** GCG style token optimisation of the note against probe score plus a call likelihood term.
3. **White box embedding attack.** Upper bound, as in Bailey et al.

Then compare probe designs under the same attacks: last token linear, mean pooled over the note, max pooled, attention probe, an ensemble across layers, and the difference of means direction. Finally, retrain the probe on attacked notes and repeat, to show the arms race curve.

**Models.** gpt-oss-20b first, then Qwen3-8B and Llama-3.1-8B.

**Cost.** About 30 to 40 GPU hours, $150 to $180.

**Payoff.** Either a robust design for obedience monitors, or evidence that execution can be decoupled from its readable signal. Both are publishable.

### 2. Forecasting at write time: will this memory ever be acted on?

**Question.** When a memory is first written, or first retrieved, can the model's own state predict whether it will cause the planted action in a **later** session?

**Why it is open.** MemoryDojo (Microsoft, Nov 2026) measures the lifecycle (insertion, retrieval, behavioural reach) with no internals. Hidden in Memory (2026) probes detect poisoned content, not later behaviour. The NeurIPS 2026 trigger and probe protocol is behavioural only. Nobody predicts downstream behavioural reach from activations.

**Experiment.**
1. Run real memory systems (Mem0, A-MEM, Letta) with open models over multi-session episodes: plant in session 1, ordinary sessions, trigger in session N.
2. Capture activations at the write step, at each retrieval, and at the trigger.
3. Ask how early the eventual execution becomes predictable, and whether the write time signal beats a text classifier on the stored memory.
4. Include benign "conditional" memories ("when the user asks X, do Y") as hard negatives.

**Models.** gpt-oss-20b, Qwen3-8B, Qwen3-32B.

**Cost.** Mostly generation, 40 to 60 GPU hours, $180 to $260.

**Payoff.** A memory write firewall that works before the attack fires. This is the most memory specific and most novel direction.

### 3. Where does obedience come from? Tracing it through training

**Question.** Is obedience to planted notes learned in pretraining, instruction tuning, preference tuning or RL? And when a model is trained to resist injections, does the obedience representation disappear, or is it just gated?

**Why it is open.** OLMo 3 releases base, SFT, DPO and RLVR checkpoints. A July 2026 case study found SFT doubles hint following and DPO brings back concealment, but nothing has been done for injected instructions. Meta SecAlign (Llama 3.1 8B and 70B trained against injection) and its base model allow a clean comparison. No one has compared internals before and after injection defence training.

**Experiment.**
1. Execution rate and probe AUROC at each OLMo 3 stage (7B and 32B) on behavior_v1 and AgentDojo.
2. Transfer the probe trained on one stage to the others.
3. Model diffing (crosscoders) between Llama 3.1 Instruct and Meta SecAlign: does the obedience signal still exist in SecAlign but no longer drive the call?

**Cost.** 20 to 30 GPU hours, $90 to $130.

**Payoff.** A developmental story ("obedience appears at SFT, defence training gates it at layer L") that also says whether monitors survive safety training.

### 4. Does the model know it is about to obey? Self-report vs probe vs reasoning

**Question.** gpt-oss writes visible reasoning before acting. Does that reasoning mention the planted note before it executes it? Does a direct question ("Will you carry out the pending task in your memory?") predict behaviour? How do these compare with the probe, and when does each signal appear?

**Why it is open.** Dong et al. (2026) report a knowledge–action gap for exposure using reasoning. The KAIST self-report study (2026) finds models cannot recognise their own compromised outputs, but not in agents. OpenAI's chain of thought monitoring work is on reward hacking. Nobody puts a probe, a reasoning monitor and self-report side by side for execution in agents.

**Experiment.**
1. **Free, on the CPU, now:** a reasoning monitor on the saved run 3 text (the analysis channel is in the saved generations).
2. **Then on the GPU:** self-report questions branched from the same prompt, plus timing (which signal fires first).

**Cost.** First part free, second part about 5 GPU hours, $25.

**Payoff.** Practical advice on which monitor to deploy, and a test of whether models have access to their own pending decision.

### 5. Finding the mechanism with matched-length notes and transcoders

**Question.** Where exactly does the reply read the note? This resolves our disagreement with Readable Before Actionable.

**Experiment.**
1. Build behavior_v2 with minimal pairs that differ in **one token** and have the same length ("still pending" vs "already completed"), so note tokens can be patched position by position.
2. Use attention head patching and path patching from the reply to the note.
3. Use Qwen3 cross-layer transcoders (released 2026, usable with circuit-tracer and Neuronpedia) for attribution graphs.

This also answers the "one word changes the answer" question directly.

**Cost.** 20 GPU hours, $90.

**Payoff.** A circuit level account of obedience, which is what interpretability venues reward.

### 6. Contagion: one poisoned memory in a group of agents

**Question.** In a group of agents sharing memory, does one planted note spread like an epidemic? Can probe based quarantine at hub agents stop it more cheaply than random monitoring?

**Why it is open.** Prompt Infection (2024) shows logistic spread in agent societies (full infection by 21 to 47% of turns) with prompt level defences only. No one uses activation monitors as a network defence.

**Experiment.**
1. Measure per agent transmission (execute and rewrite) on gpt-oss-20b.
2. Simulate 50 to 500 agents on different network shapes.
3. Compare probe gated quarantine strategies.

**Cost.** 30 to 50 GPU hours, $130 to $220. This is the SFI proposal's question.

**Payoff.** A system level safety result and a bridge to complex systems.

### 7. Generality and credibility (needed for any main conference paper)

* **Models.**
  * gpt-oss-20b and gpt-oss-120b (scale within one family).
  * Qwen3 8B, 14B and 32B (has transcoders).
  * Llama 3.1 8B and Meta SecAlign 8B.
  * Gemma 3 12B or Gemma 4 (has Gemma Scope SAEs).
  * OLMo 3 7B (fully open data).
* **Benchmarks.**
  * behavior_v1, plus AgentDojo (the standard), InjecAgent.
  * AgentDyn (dynamic, 2026), LLMail-Inject (real human attacks).
  * MCPTox (tool poisoning), MemoryDojo (memory lifecycle).
* **Baselines.**
  * A fine tuned small text classifier (ModernBERT or Qwen3 0.6B) on the full prompt.
  * The conversation statistics baseline of Kale and Vishwanath.
  * An LLM judge (Jev v2, a frontier model).
  * Existing detectors (PIShield, Attention Tracker, Dong probe, role probes).
* **Cost.** $150 to $250 across models.

## A suggested programme

* **Quick wins** (this week, under $50): direction 4's free part on the saved text, stronger text baselines on the CPU, and a steering rerun with text saved to settle "breaks tool use".
* **Flagship** (December to January): directions 1 and 3 together. Asking whether obedience monitors survive both adaptive attackers and safety training is one coherent "probe robustness" paper, squarely in the fellowship's topic. About $300.
* **Stretch** (if results are strong): direction 2 as a second paper, or direction 6 for the SFI connection.

Total for quick wins plus flagship plus generality: about $500 to $600, leaving a reserve.

## Sources

Obfuscated Activations Bypass LLM Latent-Space Defenses (Bailey et al., arXiv 2412.09565, ICLR 2026 per proceedings listing); RL-Obfuscation (Gupta and Jenner, arXiv 2506.14261); The Obfuscation Atlas (Taufeeque et al., FAR.AI, arXiv 2602.15515); Where does hint-following and concealment arise? OLMo-3 checkpoints (LessWrong, July 2026); Tracing eval-awareness emergence through training of OLMo 3 (LessWrong); Meta SecAlign (arXiv 2507.02735); SecAlign (arXiv 2410.05451); Monitoring Reasoning Models for Misbehavior (OpenAI); Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment (arXiv 2611.07586); Can LLMs Reliably Self-Report Adversarial Prefills (KAIST, arXiv 2606.23671); Your Agentic LLMs Secretly Encode IPI Exposure (arXiv 2608.02657); MemoryDojo (arXiv 2611.09721); Hidden in Memory (arXiv 2605.15338); Evaluating Memory Safety as a Trajectory: A Trigger–Probe Protocol (NeurIPS 2026); Remembering More, Risking More (arXiv 2605.17830); What Happens Inside Agent Memory? (arXiv 2605.03354); Circuit tracing with Qwen3 cross-layer transcoders (BluelightAI); circuit-tracer (github.com/decoderesearch/circuit-tracer); Prompt Infection (arXiv 2410.07283); persona vectors and the assistant axis (Anthropic); AgentDyn (arXiv 2602.03117); LLMail-Inject (arXiv 2506.09956); plus everything in `LITERATURE_REVIEW.md`.
