# Sources and citation notes

The paper is [API-TAIL, arXiv:2610.36801](https://arxiv.org/abs/2610.36801).
Its registered authors, in order, are Haocheng Mei, Yuchuan Tian, Ye Yuan, Chao Xu, Allen Chen, Kai Han and Yunhe Wang. The preferred paper citation in `CITATION.cff` and both README files follows that record. The software author and copyright holder remain TokenRhythm Technologies.

These bibliography and attribution clarifications do not change the scoring functions, audit samplers, thresholds, frozen statistics or experimental results. Updating this package or the local paper sources does not update the arXiv submission.

## Baseline sources and adaptations

- [MET, Gao et al., ICLR 2025](https://arxiv.org/html/2410.20247v2): the original work already considers a composite null covering FP32 and FP16 references. Here, matched-budget MET and the variant with native-style sampling use this study's composite calibration. They are not complete reproductions of the original evaluation protocol.
- [RUT, Zhu et al., 2025](https://arxiv.org/html/2506.06975v1): one audited API response and local reference responses per prompt. This implementation retains randomized ranks and the Cramer-von Mises test, with token-level midranks, a finite-reference rank adjustment and composite calibration as described in [DESIGN_MAP.md](DESIGN_MAP.md).
- [Fast-DetectGPT, Bao et al., ICLR 2024](https://arxiv.org/html/2310.05130v2): when the scoring and sampling distributions match the same model at temperature 1, the theoretical API-TAIL score is the negative of its standardized criterion, apart from numerical variance floors. Our fixed-temperature control uses the opposite score tail, a batch count and a composite cutoff. Historical identifiers such as `FDG-fsv` remain for archive compatibility.
- The NLL, log-rank and window-entropy counts are feature controls inspired by statistical text detection. They are not full implementations of [GLTR](https://aclanthology.org/P19-3019/), [DetectGPT](https://proceedings.mlr.press/v202/mitchell23a.html), [DetectLLM](https://aclanthology.org/2023.findings-emnlp.827/) or [Binoculars](https://proceedings.mlr.press/v235/hans24a.html). GLTR includes a visualization tool for human detection; Binoculars uses a ratio of log-perplexity to log-cross-perplexity.

For sampling budgets, score conventions, separate thresholds and the nucleus support/position changes, see [the implementation map](DESIGN_MAP.md). The MET comparison matches the API request budget, not the identical one-response-per-prompt draw. Results for these adapted baselines should not be attributed to the original methods under different null hypotheses or calibration rules.

## Bibliographic corrections and additions

The labels below identify references by author and title rather than by bibliography number.

| Reference | Verified bibliographic information | Scope of revision |
| --- | --- | --- |
| Finlayson, Ren and Swayamdipta (2024), *Logits of API-Protected LLMs Leak Proprietary Information* | **The First Conference on Language Modeling (COLM 2024)**. [Official paper list](https://2024.colmweb.org/AcceptedPapers.html) | Corrects the mistaken ICML venue. |
| Gubri et al. (2024), *TRAP: Targeted Random Adversarial Prompt Honeypot for Black-Box Identification* | Findings of ACL 2024, **11496-11517**. [ACL Anthology](https://aclanthology.org/2024.findings-acl.683/) | Corrects the page range. |
| Bruckner (2026), *One Token Is Enough* | **Tomáš Bruckner**. [Paper byline](https://arxiv.org/html/2607.10252v1) | Restores the author's diacritics; the arXiv metadata also uses Tomas. |
| Chauvin et al. (2026), *Token-Efficient Change Detection in LLM APIs* | ICML 2026, **PMLR 306:13247-13277**. [Proceedings](https://proceedings.mlr.press/v306/chauvin26a.html) | Adds the series and volume; existing pages were correct. |
| Hans et al. (2024), *Spotting LLMs With Binoculars* | ICML 2024, **PMLR 235:17519-17537**. [Proceedings](https://proceedings.mlr.press/v235/hans24a.html) | Adds volume and pages. |
| Lin et al. (2026), *Behavioral Consistency and Transparency Analysis on Large Language Model API Gateways* | IMC 2026, **563-573**, DOI **10.1145/3777912.3809156**. [Publisher record](https://doi.org/10.1145/3777912.3809156) | Adds pages and DOI; the venue was correct. |
| Mitchell et al. (2023), *DetectGPT* | ICML 2023, **PMLR 202:24950-24962**. [Proceedings](https://proceedings.mlr.press/v202/mitchell23a.html) | Adds volume and pages. |
| Richardeau et al. (2026), *FLIPS* | ICML 2026, **PMLR 306:38974-38993**. [Proceedings](https://proceedings.mlr.press/v306/gurvan26a.html) | Adds the series and volume; retains **Gurvan Richardeau**, matching the [paper byline](https://arxiv.org/html/2606.03330v1). |
| Sun et al. (2025), *Idiosyncrasies in Large Language Models* | ICML 2025, **PMLR 267:57854-57885**. [Proceedings](https://proceedings.mlr.press/v267/sun25z.html) | Adds volume and pages. |
| Zhang et al. (2026), *Real Money, Fake Models: Deceptive Model Claims in Shadow APIs* | **Accepted at CCS 2026**. [Official accepted papers](https://www.sigsac.org/ccs/CCS2026/program/accepted-papers.html) | States accepted/to-appear status as of the manuscript's 27 October 2026 date; no unverified pages or DOI are added. |

The [GPTQ arXiv title](https://arxiv.org/abs/2210.17323) is retained. GPTQ/OPTQ title differences across versions do not indicate a fictitious reference. The FLIPS author's given and family names are not reversed to match the inconsistent proceedings metadata.

The paper also separates commercial-gateway observations from [Pape et al.'s controlled inference-backend study](https://arxiv.org/html/2605.19537v2). It acknowledges MET's earlier composite null and distinguishes RUT's local reference generation from repeated target-API queries.
