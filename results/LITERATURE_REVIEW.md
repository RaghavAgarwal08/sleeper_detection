# Literature review and novelty check

November 24, 2026. Covers work up to late November 2026 found through arXiv, the ICML, NeurIPS, ICLR, ACL and NAACL proceedings, and company research pages (Microsoft, Google DeepMind, OpenAI, Anthropic, Apple). For the most relevant papers the full text was read; where only part of a long paper could be read, the table says so. Every claim about a paper below comes from that paper's own text. AI assistance (Claude) was used to search and summarize; the verification column says how much of each paper was read.

## Bottom line

The field moved fast in the last three months. Four of our five findings now have close prior work, much of it posted between August and October 2026.

| Our finding | Closest prior work | Novelty |
| --- | --- | --- |
| A probe predicts whether the agent will **execute** a planted instruction, beating a text baseline within wording, pre-registered | Readable Before Actionable (Oct 2026), Role Confusion (ICML 2026), Operant AI (Oct 2026), Heo et al. (ICLR 2025) | **Partly new.** Predicting compliance from internals has been done. What is new is the measurement discipline (within wording AUROC, full prompt text baseline, conditional probing, pre-registration) and the memory note channel. It also contradicts CounterSteer's claim that the decision is not readable before generation. |
| Attention based detectors are at chance for execution; the generic yes/no probe does not transfer | Operant AI (Oct 2026) on presence vs influence | **Mostly new as a result on gpt-oss-20b**, but a small contribution. |
| The probe's position is not on the causal path; the probe direction is a poor lever | Readable Before Actionable, Where Do LLMs Decide (Sep 2026), Lacuna (ICML 2026 workshop), CounterSteer | **Not new as a principle.** Our clean patching null is a useful data point, and it disagrees with one published patching result (see below). |
| A difference of means direction (executed minus resisted) suppresses execution while tool use survives | **CounterSteer (Microsoft, Sep 2026)**, Readable Before Actionable, Lacuna, ARGUS | **Not new.** CounterSteer does almost exactly this, on gpt-oss-20b among others, with held-out attacks, utility and adaptive attackers. |
| Everything is on poisoned **memory** notes | MemoryDojo (Microsoft, Nov 2026), Hidden in Memory (May 2026), MemPoison (NeurIPS 2026) | The memory channel itself is crowded, but **no paper we found predicts or steers execution of memory planted instructions from activations with surface controls**. |

**Recommendation.** Do not lead the paper with the steering defense. Lead with the measurement problem the field is now worried about: probes for prompt injection often learn wording and conversation shape rather than intent (Kale and Vishwanath 2026; Amari et al. 2026; the ICML 2026 paper on probes relying on textual evidence). Our within wording evaluation against a full prompt text baseline is a direct answer to that worry, applied to the harder question of predicting obedience rather than presence. The steering results become a replication and extension of CounterSteer on the memory channel, with a clean comparison of probe vs difference of means directions and size matched random controls.

## 1. Predicting execution (compliance) from internal states

**What exists.**

* **Heo et al., "Do LLMs know internally when they follow instructions?" (Apple, ICLR 2025)** [2410.14516]. Linear probes on the first input token predict whether a simple instruction will be followed: AUROC 0.74 to 0.88 on unseen tasks, but 0.50 to 0.56 on unseen instruction types. Steering along the probe direction raises success slightly without hurting quality. Not about injection.
* **Ye, Cui and Hadfield-Menell, "Prompt Injection as Role Confusion" (ICML 2026)** [2603.12277]. Role probes ("userness", "CoT-ness") trained on identical content with different role tags predict injection success before generation, on gpt-oss-20b and others. Agent injections succeed 0 to 2% on most models but 26% on gpt-oss-20b. Strong surface controls (destyling, retagging).
* **Yu et al., "Readable Before Actionable" (Zhejiang, Oct 2026)** [2610.05295]. A probe for instruction vs data role (AUROC about 0.95, structural features only 0.53) applied without retraining to AgentDojo hijacked vs resisted trajectories gives 0.82 to 0.84 at mid to late layers. Qwen, Llama, Gemma. *Partly read.*
* **Gupta and Tembey, "Are You Reading the Right Thing?" (Operant AI, Oct 2026, workshop)** [2610.19452]. Compares five internal detectors on successful vs unsuccessful attacks against template matched benign controls. Main point: "detecting prompt injection and detecting its behavioral influence are different problems". Presence detectors rank failed attacks higher; signals from response formation rank successful ones higher. Qwen3.6 and Gemma 4, not gpt-oss.
* **Dong et al., "Your Agentic LLMs Secretly Encode Indirect Prompt-Injection Exposure in Hidden States" (Tsinghua, Aug 2026)** [2608.02657]. Probes on eight models including gpt-oss-20b reach 0.93 to 0.98 AUROC on held out AgentDojo suites, but the label is **exposure** (an injection is present), explicitly "independently of whether the model is ultimately compromised". No text baseline in the part we read.
* **CounterSteer (Russinovich, Microsoft, Sep 2026)** [2609.36570] reports the opposite of what we find: "the decision becomes linearly readable at argument emission but not at the examined pre-generation sites".

**How we differ.**

1. We predict execution measured from 8 sampled rollouts per note (an execution rate), not a single outcome.
2. We compare only notes with the same wording and beat a TF-IDF reader of the **whole prompt** (+0.10 on held out cases, +0.14 on held out domains). Conditional probing shows activations add 0.035 bits beyond wording and text, and text adds nothing beyond activations. None of the papers above reports a within template comparison against a full prompt text model for the execution label.
3. The tests were pre-registered.
4. Our pre-generation readout at the end of the memory result (0.88 to 0.90 within wording) directly conflicts with CounterSteer's negative finding. That conflict is worth stating and explaining. Possible reasons are our position choice (the note's last token), the memory channel, or our within wording metric.

**Verdict: partly new.** The question has been asked; our answer under strict surface controls has not.

## 2. Surface confounds in probes (why our methodology matters now)

* **Kale and Vishwanath, "Prompt-Injection Probes Learn the Conversation, Not Just the Injection" (Georgia Tech, Oct 2026)** [2610.18408]. On six models including gpt-oss-20b, standard injection probes flag 70% of clean twins and 94% of harmless look-alikes in new applications. A 7 feature conversation statistics baseline reaches 0.96 to 0.98 AUROC in seen applications.
* **Amari, Noël and Capo-Chichi, "The Illusion of Intent: Linear Probes as Sophisticated Keyword Counters in LLM Safety Detection" (ICML 2026 workshop)**. AUROC "drops from 1.0 to below 0.10 after lexical ablation". *Abstract only.*
* **"Linear probes rely on textual evidence: Results from leakage mitigation studies in language models" (ICML 2026)** [2509.21344]. *Title and listing only.*
* **"When AUC 0.998 Is Not Enough: A Candidate Evaluation Protocol for Hidden-State Probes of Indirect Prompt Injection in Multimodal Computer-Use Agents" (2026)** [2606.22864]. *Listing only.*
* Older background: control tasks for probes (Hewitt and Liang 2019) and conditional probing (Hewitt et al. 2021).

**What this means for us.** Our benign twins, within wording AUROC, full prompt TF-IDF baseline and conditional probing are exactly the controls these papers say are missing. This is the strongest framing for the paper.

## 3. Attention based detectors

* **Attention Tracker** (Hung et al., NAACL Findings 2025) [2411.00348], **Instruction Detection** (Wen et al., EMNLP Findings 2025) [2505.06311, hidden states plus gradients, 96.9% out of domain presence detection], **TaskTracker** (Abdelnabi et al., SaTML 2025) [2406.00799, activation deltas for task drift], **AlignSentinel** [2602.13597], and **"Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs"** [2512.08417] all target **presence**.
* Operant AI (above) reports Attention Tracker ranks successful attacks above unsuccessful ones on Qwen (0.949 vs 0.691 against benign controls). That is a different comparison from ours (successful vs resisted within wording).

**Our result.** On gpt-oss-20b, within wording, Attention Tracker and AttnDA score 0.49 for execution, AttentionDefense 0.67, the probe 0.88. A plausible reason specific to this model is that half its layers only attend 128 tokens back. This evaluation on gpt-oss-20b appears to be new, but it is a supporting result, not a headline.

**Generic yes/no probe.** MacDiarmid et al. (Anthropic, 2024) showed generic "are you doing something dangerous" directions catch sleeper agents. We found no paper testing this on prompt or memory injection; our negative result (0.52 to 0.57) is probably new and worth a paragraph.

## 4. Readable vs causal

* **Readable Before Actionable** [2610.05295] names a "readability–actionability gap": role is decodable well before edits change the tool choice. However, its full state patching at the final pre-action token **does** flip decisions around layers 16 to 22, and edits at pre-action and injected span positions cut attack success by about 53 to 56 points on AgentDojo (summary level reading).
* **Wen et al., "Where Do LLMs Decide to Break the Rules?" (RIKEN and others, Sep 2026)** [2609.37737]. Attack information is decodable from layer 1, but patching only gains leverage late (peak around 67% depth), flipping 77 to 92% of compliant runs. Mostly chatbot role conflict, not agents.
* **Zhuravel, Pysklo and Au-Yeung, "Activation Steering for Tool-Poisoning Defense in Language-Model Agents" (ICML 2026 AIWILD workshop)**. Later layers "might be great at detecting" tool poisoning but are "useless for steering" (Qwen3-8B). *Read through an aggregator page only.*
* **CounterSteer**: "A direction that correlates but does not steer is rejected."

**Our result and how it differs.** Patching the last 4 prompt tokens (end of the memory result through the token before the reply) with the benign twin's state at layers 12, 17 or 22 changes execution by at most 0.013. The benign twin's own prompt gets 0.00. So on gpt-oss-20b, in the memory channel, the reply reads the decision from the note's tokens, not from a summary at the end. This **disagrees** with Readable Before Actionable's pre-action patching flips (Qwen, Llama). Reasons to check are the model (gpt-oss is a mixture of experts with sliding window layers), the channel, and that they patch along a role direction at the final token while we swap the full state at 4 positions. That disagreement is a concrete, publishable point.

## 5. Steering as a defense

* **CounterSteer (Mark Russinovich, Microsoft Azure, arXiv 29 Sep 2026)** [2609.36570]. Very close to ours.
  * The direction is "the difference-in-means direction between followed- and resisted-injection sides, per layer", built from paired episodes that differ only in whether the injection is followed. They state "We prefer difference-in-means over probe weights".
  * It is subtracted on every tool-result token during prefill.
  * Five models including **gpt-oss-20b** (layers 12, 16, 20): AgentDojo compromise 0.475 to 0.079 at 94.4% benign utility (summary level for the per model numbers).
  * Pre-specified held-out evaluation, LLMail-Inject replay (none of 2,052 succeed), adaptive and white-box attackers. Parameter manipulation is the main weakness.
* **Readable Before Actionable**: hijacked vs resisted directions cut attack success at pre-action and injected span positions, with benign utility about 95%.
* **Lacuna workshop paper**: difference of means (resisted minus complied) on Qwen3-8B raises the defense rate from 54% to 82% on MCPTox; over-steering collapses tool use.
* **ARGUS** (Lu et al., SCUT, Dec 2025) [2512.05745]: probe weight directions, gated by a detector, for multimodal injection; attacker instruction accuracy to about 0.
* Background: refusal is mediated by a single direction, found by difference of means (Arditi et al. 2024) [2406.11717]; mass mean probes are more causal than logistic regression probes (Marks and Tegmark, "The Geometry of Truth") [2310.06824]; difference of means is a strong simple baseline for concept directions (AxBench, ICML 2025) [2501.17148]; refusal is more than a single direction [2602.02132]; representation engineering and circuit breakers (Zou et al. 2023, 2024); training-time defenses (instruction hierarchy, StruQ, SecAlign); system-level defenses (CaMeL, AgentSentry [2602.22724], AttriGuard [2603.10749]); and steering vectors as an attack surface [2606.05958].

**Verdict: not new as a method.** Our run 2 finding (difference of means −1 cuts execution 0.26 to 0.07 with tool use at 0.91, beating size matched random pushes) independently replicates CounterSteer's core idea in a different channel.

What we can still add:

1. The memory note channel.
2. A direct, controlled comparison showing the probe direction breaks tool calling at the same doses where the difference of means direction is selective. CounterSteer states its preference but does not, in what we read, show this.
3. Size matched random controls.
4. Run 3's held out notes plus a utility measure on clean prompts.

Run 3 should also add CounterSteer's recipe (subtract on note tokens only) as a comparison, so we are measured against the current state of the art.

## 6. Memory poisoning (our channel)

* **MemoryDojo: Benchmarking Poisoning Through LLM Agent Memory** (Yang, Bullwinkel, Minnich, Amiri, Severi; Microsoft and RPI; arXiv 2611.09721). Our terminology (lifecycle states, poisoned note) follows this benchmark, and it should be cited. It evaluates five memory systems and no defenses, with no internal state methods.
* **Hidden in Memory: Sleeper Memory Poisoning in LLM Agents** (Pulipaka et al., with Abdelnabi and Fritz; May 2026, under review) [2605.15338]. Includes activation probes (0.93 to 0.99 AUROC) on models including GPT-OSS-20B, but they detect the adversarial content, not whether the agent will act.
* Others, all text or system level with no activation work on execution:
  * From Untrusted Input to Trusted Memory (Dash et al.) [2606.04329]
  * MemPoison (NeurIPS 2026) [2607.14651]
  * Bad Memory [2607.14611]
  * MemSecBench [2607.27080]
  * MemForm [2610.14431]
  * Forged Reasoning Attacks [2607.05029]
  * MemSentry [2609.08747, text classifiers]
  * A-MemGuard (ICML 2026) [2510.02373]
  * Memory Poisoning Attack and Defense [2601.05504]
  * MINJA [2503.03704]
  * AgentPoison (NeurIPS 2024)
  * a survey [2604.16548]

**Verdict.** The gap we fill is real and narrow: **activation level prediction and suppression of execution for memory planted instructions, with surface controls**.

## 7. Industry practice

Google DeepMind describes production probes for Gemini [2601.11516] and OpenAI describes training activation classifiers for safety. *Titles and listings only.* Neither, in what we found, targets execution of memory injected instructions. Anthropic's sleeper agent probes (MacDiarmid et al. 2024) are the origin of our generic probe test.

## 8. What to do with this

1. **Reframe the paper** around measuring obedience rather than presence, under the surface controls the 2026 confound papers call for. The memory channel is the setting, and gpt-oss-20b is a model several of these papers also use, which makes comparison easy.
2. **State the two disagreements openly.** First, CounterSteer finds the decision not readable before generation, while we read it at 0.88 to 0.90 within wording. Second, Readable Before Actionable flips decisions by patching at the pre-action token, while our patching at the last 4 tokens does nothing on gpt-oss-20b. Disagreements with recent, prominent preprints are what reviewers find interesting.
3. **Position steering as a replication and extension of CounterSteer,** not a new defense. Add CounterSteer style steering (subtract on note tokens only, during prefill) as a condition in a later run.
4. **Add a stronger text baseline** (a fine tuned small model reading the prompt) and the conversation statistics baseline of Kale and Vishwanath. Reviewers who know that paper will ask for it.
5. **Multi-model.** Most competing papers use 4 to 8 models; one model will not be enough for a main conference.
6. **Cite MemoryDojo** as the source of the terminology.

## References (arXiv identifiers link to arxiv.org/abs/ID)

* Abdelnabi et al. 2025. Get my drift? Catching LLM task drift with activation deltas. SaTML. 2406.00799
* Amari, Noël, Capo-Chichi 2026. The illusion of intent: linear probes as sophisticated keyword counters in LLM safety detection. ICML Hypothesis Testing Workshop. https://icml.cc/virtual/2026/79985
* Arditi et al. 2024. Refusal in language models is mediated by a single direction. 2406.11717
* Dash et al. 2026. From untrusted input to trusted memory. 2606.04329
* Dong et al. 2026. Your agentic LLMs secretly encode indirect prompt-injection exposure in hidden states. 2608.02657
* Gupta, Tembey 2026. Are you reading the right thing? Rethinking internal signals for prompt-injection detection. 2610.19452
* Hazimeh et al. 2026. Beyond the prompt: leveraging pre-decoding states for jailbreak detection in dLLMs. 2611.10258
* Heo et al. 2025. Do LLMs "know" internally when they follow instructions? ICLR. 2410.14516
* Hung et al. 2025. Attention Tracker: detecting prompt injection attacks in LLMs. NAACL Findings. 2411.00348
* Kale, Vishwanath 2026. Prompt-injection probes learn the conversation, not just the injection. 2610.18408
* Lu et al. 2025. ARGUS: defending against multimodal indirect prompt injection via steering instruction-following behavior. 2512.05745
* MacDiarmid et al. 2024. Simple probes can catch sleeper agents. Anthropic. https://www.anthropic.com/research/probes-catch-sleeper-agents
* Marks, Tegmark 2024. The geometry of truth. 2310.06824
* Pulipaka et al. 2026. Hidden in memory: sleeper memory poisoning in LLM agents. 2605.15338
* Roy, Basu 2026. MemSentry. 2609.08747
* Russinovich 2026. CounterSteer: suppressing indirect prompt injection with activation steering. 2609.36570
* Wei et al. 2026. A-MemGuard. ICML. 2510.02373
* Wen et al. 2025. Defending against indirect prompt injection by instruction detection. EMNLP Findings. 2505.06311
* Wen et al. 2026. Where do LLMs decide to break the rules? 2609.37737
* Wu et al. 2025. AxBench. ICML. 2501.17148
* Yang et al. 2026. MemoryDojo: benchmarking poisoning through LLM agent memory. 2611.09721
* Ye, Cui, Hadfield-Menell 2026. Prompt injection as role confusion. ICML. 2603.12277
* Yu et al. 2026. Readable before actionable: causal tracing of indirect prompt injection. 2610.05295
* Zhuravel, Pysklo, Au-Yeung 2026. Activation steering for tool-poisoning defense in language-model agents. ICML AIWILD workshop. https://openreview.net/forum?id=duDmnRyJh5
* Zverev et al. 2025. Can LLMs separate instructions from data? ICLR. 2403.06833

## Limits of this review

* Several long papers could only be read in part. Numbers marked "summary level" came from automated extraction of the unread parts and should be checked against the PDFs before they are cited in the paper.
* Search was through web search and arXiv, not a database query with a fixed protocol, so a relevant paper could have been missed. The weekly volume of arXiv preprints in this area is high.
* Venues were taken from the papers' own pages or the conference sites; preprints without a stated venue are listed as preprints.
