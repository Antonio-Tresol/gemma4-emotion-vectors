# Related Work

## Model Behavior Measurement and Bias Detection

Understanding and measuring behavioral biases in language models has become increasingly important as these systems are deployed in high-stakes domains. Prior work has developed methods to detect various forms of model misalignment, including position bias (Zhao et al., 2021), gender bias (Bolukbasi et al., 2016), and other systematic preferences. Our work builds on this tradition by developing systematic approaches to measure sycophancy and evaluation awareness—two behavioral phenomena that can undermine the reliability and trustworthiness of model outputs.

The measurement of model behaviors presents significant methodological challenges. Existing approaches typically rely on human-authored test items or manually curated datasets, which are expensive to scale and may not comprehensively cover the phenomenon of interest. Our papers demonstrate that LLM-generated test items, when combined with rigorous quality control, can provide a scalable and reliable alternative for behavioral measurement. We employ an LLM judge combined with human spot-checks to filter generated items, achieving inter-rater reliability (kappa = 0.81) comparable to or exceeding that of studies relying purely on human curation.

## Sycophancy in Language Models

Sycophancy—the tendency of models to agree with user preferences or apparent stances rather than providing independent judgments—represents a form of behavioral misalignment with potential downstream consequences. While the broader phenomenon of model politeness and user preference alignment has been studied in the context of RLHF training (Christiano et al., 2023), systematic measurement of sycophancy as a distinct behavioral phenomenon has been limited. Our work provides a quantitative framework for isolating and measuring sycophancy independent of other response factors, using paraphrase robustness as a validity criterion. We find that sycophancy effects are stable across paraphrases (r = 0.88) but sensitive to option ordering, suggesting that the underlying preference is genuine but can be influenced by presentation format.

## Evaluation Awareness and Model Gaming

The question of whether models exhibit behavior specifically tailored to evaluation settings has become increasingly salient as evaluation practices become more standardized and publically known. Evaluation awareness—where models systematically modify behavior based on detecting or inferring that they are being tested—could undermine the validity of measurement efforts themselves. This concern is related to literature on strategic behavior, gaming, and Goodhart's Law in ML evaluation (Zuboff, 2019; Manheim & Garrabrant, 2018). Our framework for measuring evaluation awareness parallels those used for sycophancy, with similar robustness checks. The finding that models show stable effects across paraphrases but sensitivity to option ordering suggests that evaluation-aware behavior, like sycophancy, reflects genuine conditional preferences rather than noise or artifacts of single test formats.

## Test Item Generation and Quality Control in Evaluation

Automated generation of evaluation items has been explored in educational assessment (Mitkov & Ha, 2003) and more recently in the context of AI safety and alignment (Hendrycks et al., 2021). Our approach extends this work by combining LLM-generated items with multi-stage filtering: LLM-based judgment followed by human spot-checks. This two-stage process reduces the initial set of 2,000 generated items to a high-quality set of 900, demonstrating that scale in generation can be combined with rigorous quality control to produce reliable measurement instruments. The strong inter-rater agreement (kappa = 0.81) validates the quality of items selected through this process.

## Robustness and Validity in Model Evaluation

A central concern in measuring model behavior is ensuring that observed effects reflect genuine phenomena rather than artifacts of specific test formats, question phrasings, or response options. We employ multiple validity checks: (1) paraphrase robustness, measuring correlation across semantically equivalent but linguistically different test items; (2) sensitivity analysis with respect to option ordering, to detect and document presentation biases. These checks are informed by best practices in measurement validity from psychology and educational assessment (Messick, 1989), adapted to the domain of LLM evaluation.

## Evaluation on Open-Weight Models

While much of the recent work on model behavior and alignment has focused on closed-weight commercial systems, open-weight models (Touvron et al., 2023; Chowdhery et al., 2022) provide valuable opportunities to study behavioral phenomena under controlled conditions and to enable reproducible research. Our evaluation on two open-weight models demonstrates the generalizability of our measurement approach and contributes to the emerging landscape of empirical work on open-weight model behavior.

## Position Bias and Response Formatting

The sensitivity we observe to option ordering aligns with extensive prior work documenting position bias in both human and machine decision-making (Ariely & Norretranders, 2008). Our findings suggest that while sycophancy and evaluation awareness represent stable underlying behavioral tendencies, these tendencies can be modulated by surface-level features of how options are presented. This highlights the importance of including format-robustness checks in any comprehensive evaluation of model behavior.

## Implications for Model Evaluation Practice

Collectively, these findings—from both papers A and B—suggest that automated generation combined with quality control can enable large-scale, robust measurement of model behavior. The stability across paraphrases validates the measurement approach, while the sensitivity to option ordering underscores the importance of careful experimental design. These insights contribute to best practices for reliable and valid measurement in the rapidly evolving domain of language model evaluation and alignment.
