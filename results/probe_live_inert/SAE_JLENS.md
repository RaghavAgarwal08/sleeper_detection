# SAE and Jacobian lens readouts on behavior_v1 (exploratory, Neuronpedia)

Script: `interp/probe_live_inert/sae_jlens_analysis.py`; data collected with `neuronpedia_collect.js`. SAE sample: 48 live/inert pairs and 70 live notes for the fired contrast. J-lens sample: 48 pairs and 70 live notes.
Small samples, many features: treat single latents as leads to check with steering, not findings. q = Benjamini-Hochberg.
Explanations are Neuronpedia's automatic labels. At this position (a closing brace) nearly all of them describe the local token or format (punctuation, braces, code), so the latents that separate live from inert are not interpretable as status features from their labels alone.

## SAE layer 15, end of the memory result

Live vs inert: 92 latents tested, 53 with q < 0.05. Fired vs not fired within wording: 87 tested, 0 with q < 0.05.

Decoding from SAE latents vs the dense probe on the same notes (grouped CV by case): live vs inert {'sae_l1': {'auroc': 0.969, 'auroc_within_stratum': 0.97}, 'sae_l2': {'auroc': 0.999, 'auroc_within_stratum': 0.999}, 'dense_probe': {'auroc': 1.0, 'auroc_within_stratum': 1.0}}, fired {'sae_l1': {'auroc': 0.5, 'auroc_within_stratum': 0.5}, 'sae_l2': {'auroc': 0.456, 'auroc_within_stratum': 0.461}, 'dense_probe': {'auroc': 0.799, 'auroc_within_stratum': 0.763}}.

| Higher on live notes | pairs live > inert | q | explanation |
| --- | --- | --- | --- |
| 21057 | 45 of 48 | 1.01e-07 | Code snippets (active on 0.01% of tokens) |
| 6919 | 46 of 48 | 1.27e-07 | : (active on 0.18% of tokens) |
| 115053 | 42 of 48 | 3.02e-07 | itive (active on 9.4% of tokens) |
| 91358 | 43 of 48 | 5.33e-07 | code and math (active on 41% of tokens) |
| 2223 | 41 of 48 | 5.33e-07 | news articles (active on 0.02% of tokens) |
| 128114 | 44 of 48 | 9.52e-07 | `` ```` (active on 0.01% of tokens) |
| 71969 | 40 of 48 | 1.62e-06 | Period (active on 1.3% of tokens) |
| 70456 | 38 of 48 | 2.04e-05 | </end/> (active on 0.03% of tokens) |
| 28034 | 38 of 48 | 6.02e-05 | code and/or lists (active on 1.1% of tokens) |
| 29766 | 35 of 48 | 0.000158 | Code errors (active on 0.01% of tokens) |

| Higher on inert notes | pairs inert > live | q | explanation |
| --- | --- | --- | --- |
| 98322 | 36 of 48 | 3.46e-08 | Code/math symbols (active on 32% of tokens) |
| 91476 | 48 of 48 | 7.45e-08 | Unstructured computer data (active on 0.52% of tokens) |
| 54891 | 38 of 48 | 9.52e-07 | punctuation (active on 0.13% of tokens) |
| 82092 | 40 of 48 | 1.7e-06 | say "User" (active on 0.01% of tokens) |
| 104543 | 38 of 48 | 2.73e-06 | ``` (active on 0% of tokens) |
| 89425 | 35 of 48 | 4.46e-06 | Incomplete text generation (active on 0% of tokens) |
| 54250 | 32 of 48 | 5.85e-06 | mixed text types (active on 0.71% of tokens) |
| 122967 | 35 of 48 | 6.81e-06 | } (active on 0.03% of tokens) |
| 17858 | 29 of 48 | 1.28e-05 | mathematical symbols (active on 10% of tokens) |
| 70695 | 26 of 48 | 4.14e-05 | making assumptions (active on 0.02% of tokens) |

| Higher when the note fires (within wording) | AUROC | q | explanation |
| --- | --- | --- | --- |
| 119834 | 0.60 | 0.75 | markup language (active on 0.19% of tokens) |
| 70456 | 0.64 | 0.75 | </end/> (active on 0.03% of tokens) |
| 104804 | 0.62 | 0.75 | Mathematical markdown (active on 0.07% of tokens) |
| 7286 | 0.63 | 0.75 | Clear (active on 0% of tokens) |
| 17130 | 0.53 | 0.75 | end (active on 0.07% of tokens) |
| 82400 | 0.55 | 0.75 | {" (active on 0% of tokens) |
| 54891 | 0.64 | 0.75 | punctuation (active on 0.13% of tokens) |
| 104543 | 0.61 | 0.75 | ``` (active on 0% of tokens) |

| Higher when the note does not fire | AUROC | q | explanation |
| --- | --- | --- | --- |
| 99589 | 0.33 | 0.607 | "both" in different languages (active on 0.79% of tokens) |
| 23326 | 0.36 | 0.75 | / (active on 0.09% of tokens) |
| 48590 | 0.41 | 0.75 | Symbols and mixed content (active on 0.37% of tokens) |
| 28034 | 0.42 | 0.75 | code and/or lists (active on 1.1% of tokens) |
| 128114 | 0.38 | 0.75 | `` ```` (active on 0.01% of tokens) |
| 35514 | 0.42 | 0.75 | punctuation (active on 0.01% of tokens) |
| 62272 | 0.36 | 0.75 | parentheses and brackets (active on 0.02% of tokens) |
| 72802 | 0.50 | 0.75 | say "end" (active on 0.22% of tokens) |

## SAE layer 19, end of the memory result

Live vs inert: 81 latents tested, 42 with q < 0.05. Fired vs not fired within wording: 72 tested, 0 with q < 0.05.

Decoding from SAE latents vs the dense probe on the same notes (grouped CV by case): live vs inert {'sae_l1': {'auroc': 0.964, 'auroc_within_stratum': 0.967}, 'sae_l2': {'auroc': 0.938, 'auroc_within_stratum': 0.944}, 'dense_probe': {'auroc': 1.0, 'auroc_within_stratum': 1.0}}, fired {'sae_l1': {'auroc': 0.43, 'auroc_within_stratum': 0.371}, 'sae_l2': {'auroc': 0.415, 'auroc_within_stratum': 0.43}, 'dense_probe': {'auroc': 0.799, 'auroc_within_stratum': 0.763}}.

| Higher on live notes | pairs live > inert | q | explanation |
| --- | --- | --- | --- |
| 15694 | 43 of 48 | 6.05e-07 | period (active on 0.05% of tokens) |
| 29540 | 44 of 48 | 1.23e-06 | code (active on 0% of tokens) |
| 93793 | 28 of 48 | 3.79e-05 | Period (active on 0.1% of tokens) |
| 68733 | 40 of 48 | 5.08e-05 | multiple languages and code (active on 56% of tokens) |
| 49196 | 26 of 48 | 6.63e-05 | </end/> (active on 0.08% of tokens) |
| 4724 | 36 of 48 | 8.84e-05 | } (active on 0% of tokens) |
| 49776 | 37 of 48 | 0.000205 | a (active on 0.6% of tokens) |
| 36802 | 33 of 48 | 0.000383 | period (active on 0.86% of tokens) |
| 53009 | 35 of 48 | 0.000802 | excellent (active on 0.01% of tokens) |
| 4971 | 13 of 48 | 0.0011 | code symbols (active on 0.01% of tokens) |

| Higher on inert notes | pairs inert > live | q | explanation |
| --- | --- | --- | --- |
| 95087 | 48 of 48 | 1.32e-07 | Code and web structure (active on 0.22% of tokens) |
| 64122 | 44 of 48 | 8.25e-07 | . (active on 0.01% of tokens) |
| 113558 | 45 of 48 | 8.25e-07 | ] (active on 0.01% of tokens) |
| 130447 | 39 of 48 | 2.16e-06 | say ")" (active on 0.02% of tokens) |
| 94163 | 39 of 48 | 9.07e-06 | end of sequence (active on 0.13% of tokens) |
| 102117 | 35 of 48 | 9.63e-05 | punctuation (active on 0.21% of tokens) |
| 20866 | 22 of 48 | 0.00023 | no explanation on Neuronpedia |
| 24698 | 12 of 48 | 0.00208 | say dash symbol (active on 0% of tokens) |
| 12655 | 16 of 48 | 0.00392 | </channel/> (active on 0.02% of tokens) |
| 41265 | 30 of 48 | 0.00684 | ":" (active on 0.08% of tokens) |

| Higher when the note fires (within wording) | AUROC | q | explanation |
| --- | --- | --- | --- |
| 102117 | 0.63 | 0.945 | punctuation (active on 0.21% of tokens) |
| 43207 | 0.59 | 0.945 | say "202" (active on 0.01% of tokens) |
| 67194 | 0.54 | 0.945 | brackets and quotes (active on 0.03% of tokens) |
| 23396 | 0.59 | 0.945 | no explanation on Neuronpedia |
| 126647 | 0.61 | 0.945 | Code (active on 0.01% of tokens) |
| 112596 | 0.50 | 0.945 | </end/> (active on 0.01% of tokens) |
| 53009 | 0.59 | 0.945 | excellent (active on 0.01% of tokens) |
| 85149 | 0.60 | 0.945 | end token (active on 0.09% of tokens) |

| Higher when the note does not fire | AUROC | q | explanation |
| --- | --- | --- | --- |
| 58515 | 0.33 | 0.798 | } (active on 0.01% of tokens) |
| 41876 | 0.41 | 0.945 | say Model (active on 0.01% of tokens) |
| 24765 | 0.47 | 0.945 | Parenthesis (active on 0.01% of tokens) |
| 66517 | 0.44 | 0.945 | } (active on 0.02% of tokens) |
| 97385 | 0.37 | 0.945 | ) (active on 0% of tokens) |
| 89062 | 0.48 | 0.945 | Parenthesis-like characters (active on 0.02% of tokens) |
| 62626 | 0.39 | 0.945 | period (active on 0.02% of tokens) |
| 84415 | 0.43 | 0.952 | Providing (in different languages) (active on 0.04% of tokens) |

## Jacobian lens: does the planted action show up in the readout? (layers 9 to 21)

Action-word rate = fraction of workspace layers whose top 8 readout tokens include a content word of the planted action.

| Position | live | inert | pairs live > inert / inert > live | sign test p | fired | not fired | fired AUROC within wording |
| --- | --- | --- | --- | --- | --- | --- | --- |
| note | 0.016 | 0.006 | 5 / 2 | 0.453 | 0.033 | 0.026 | 0.503 |
| mre | 0.002 | 0.005 | 1 / 3 | 0.625 | 0.004 | 0.002 | 0.517 |
| last | 0.0 | 0.0 | 0 / 0 | n/a | | | |

Readout tokens at the last token of the note most enriched in live vs inert prompts (prompts with the token, live vs inert): automatically (33 vs 0), checklist (28 vs 0), pinned (27 vs 0), requester (24 vs 0), urg (18 vs 0), hints (17 vs 0), they're (15 vs 0), reminder (32 vs 2), alert (12 vs 0), task (12 vs 0), ingredient (11 vs 0), lopt (11 vs 0)

Most enriched in inert prompts: completed (39 vs 0), etc (37 vs 0), earlier (18 vs 0), disclosures (33 vs 2), canceled (13 vs 0), paragraph (11 vs 0), reviewer (45 vs 17), dialog (9 vs 0), unclear (6 vs 0), related (6 vs 0), commenters (6 vs 0), exchanges (6 vs 0)

Most enriched in fired vs not fired notes (note token): behavior (35 vs 32), sheriff's (15 vs 5), customizer (8 vs 2), instruction (27 vs 16), reminder (31 vs 23), bigger (4 vs 1), completed (17 vs 9), tweeting (21 vs 13), scroll (7 vs 3), instructions (33 vs 30), apologized (6 vs 3), tweets (17 vs 11)

Readout tokens at the last prompt token most enriched in live vs inert prompts (number of prompts): comentario (19 vs 6), apology (48 vs 47), thanking (6 vs 5), therapist (31 vs 31), concierge (48 vs 48), replied (48 vs 48), interloc (48 vs 48), commentaire (48 vs 48), chatbot (48 vs 48), gpt (48 vs 48), reply (45 vs 45), reviewer (48 vs 48)

Most enriched in inert prompts: markdown (48 vs 45), discussing (7 vs 4), apologized (26 vs 20), webhook (15 vs 11), requester (8 vs 6), replying (48 vs 48), counselor (48 vs 48), reviewer (48 vs 48), reply (45 vs 45), gpt (48 vs 48), chatbot (48 vs 48), commentaire (48 vs 48)

Most enriched in fired vs not fired (last token): comentario (15 vs 12), requester (5 vs 4), therapist (14 vs 13), concierge (35 vs 35), replied (35 vs 35), interloc (35 vs 35), commentaire (35 vs 35), chatbot (35 vs 35), gpt (35 vs 35), markdown (35 vs 35), reply (33 vs 33), reviewer (35 vs 35)

## Logit lens: does the planted action show up in the readout? (layers 9 to 21)

Action-word rate = fraction of workspace layers whose top 8 readout tokens include a content word of the planted action.

| Position | live | inert | pairs live > inert / inert > live | sign test p | fired | not fired | fired AUROC within wording |
| --- | --- | --- | --- | --- | --- | --- | --- |
| note | 0.005 | 0.002 | 2 / 1 | 1 | 0.046 | 0.0 | 0.542 |
| mre | 0.058 | 0.064 | 7 / 8 | 1 | 0.024 | 0.042 | 0.433 |
| last | 0.003 | 0.005 | 0 / 1 | 1 | 0.002 | 0.011 | 0.448 |

Readout tokens at the last token of the note most enriched in live vs inert prompts (prompts with the token, live vs inert): agenda (26 vs 0), let's (21 vs 0), amb (42 vs 5), ande (12 vs 0), wow (5 vs 0), rational (11 vs 1), ongoing (9 vs 1), rem (7 vs 1), tent (42 vs 27), pending (46 vs 39), directives (6 vs 2), ambit (36 vs 29)

Most enriched in inert prompts: previous (31 vs 0), followed (40 vs 1), led (15 vs 0), earlier (48 vs 39), printing (19 vs 1), stated (8 vs 0), lines (8 vs 0), topics (37 vs 7), jan (7 vs 0), statements (6 vs 0), ext (5 vs 0), unrelated (5 vs 0)

Most enriched in fired vs not fired notes (note token): wait (5 vs 0), off (7 vs 1), ged (10 vs 2), pending (35 vs 34), signed (6 vs 2), intended (18 vs 9), ongoing (7 vs 3), enti (11 vs 6), rem (4 vs 2), instructions (25 vs 20), ondi (15 vs 10), revis (20 vs 15)

Readout tokens at the last prompt token most enriched in live vs inert prompts (number of prompts): explanation (9 vs 0), hopefully (47 vs 39), description (9 vs 2), angle (16 vs 5), assistant (44 vs 36), nonfiction (43 vs 34), proactively (48 vs 47), formulated (48 vs 47), clarity (5 vs 2), pinned (29 vs 20), continued (19 vs 12), says (46 vs 44)

Most enriched in inert prompts: elic (48 vs 44), onward (38 vs 19), tone (48 vs 47), pun (46 vs 42), dec (12 vs 5), dialogue (36 vs 26), once (34 vs 24), language (36 vs 27), onda (15 vs 8), entes (39 vs 32), rest (36 vs 28), bas (36 vs 28)

Most enriched in fired vs not fired (last token): hopefully (34 vs 29), content (12 vs 4), language (29 vs 21), nonfiction (31 vs 27), chat (15 vs 9), pinned (26 vs 20), html (10 vs 7), using (20 vs 16), angle (11 vs 8), instruction (5 vs 4), continued (16 vs 14), helf (19 vs 17)

