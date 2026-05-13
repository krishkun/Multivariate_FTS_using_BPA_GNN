# Fuzzy BPA EGNN - Implementation Plan & Progress Tracker

**Project:** Fuzzy Basic Probability Assignment Edge Graph Neural Network for Multivariate Time Series Forecasting

**Last Updated:** 2026-05-13

---

## Thesis Structure

### Chapter 1: Introduction
- [ ] Add context for Contribution 2 (alternative evidence theory modules and explainability)
- [ ] Update objectives section
- [ ] Update research questions
- [ ] Update contributions section

### Chapter 2: Literature Review
- [ ] Add section on alternative evidence theory approaches (TBM, Pignistic, EKNN, Credal)
- [ ] Add section on explainability methods (SHAP, uncertainty quantification)
- [ ] Add relevant references for Contribution 2

### Chapter 3: Contribution 1 - Fuzzy BPA-EGNN
- [ ] Motivation
- [ ] Methodology
- [ ] Experimental Setup
- [ ] Simulation Results
- [ ] Summary

### Chapter 4: Contribution 2 - Alternative Evidence Modules & Explainability
- [ ] Motivation
- [ ] Methodology (4 alternative modules + explainability tool)
- [ ] Experimental Setup
- [ ] Simulation Results
- [ ] Summary

### Chapter 5: Conclusion
- [ ] Update with findings from both contributions

---

## Implementation Tasks

### Phase 1: Project Setup
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| Create project directory `k_zoodip/egnn/fuzzy_bpa_egnn/` | ✅ Complete | 2026-05-13 | Directory created |
| Create virtual environment `fuzzy_bpa_egnn_env` | ✅ Complete | 2026-05-13 | Environment exists |
| Create requirements.txt | ✅ Complete | 2026-05-13 | Core dependencies listed |
| Create README.md | ✅ Complete | 2026-05-13 | Documentation complete |

### Phase 2: Datasets

#### Dataset Sources and References
| Dataset | Variables | Frequency | Source | Link |
|---------|-----------|-----------|--------|------|
| Electricity | 321 | Hourly | UCI Machine Learning Repository | https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption |
| Traffic | 862 | Hourly | California Dept. of Transportation | https://pems.dot.ca.gov/ |
| Weather | 21 | 10-min | German Weather Service | https://www.bgc-jena.mpg.de/wetter/ |
| ETTh1/ETTh2 | 7 | Hourly | Electricity Transformer Temperature | https://github.com/zhouhaoyi/ETDataset |
| ETTm1/ETTm2 | 7 | 15-min | Electricity Transformer Temperature | https://github.com/zhouhaoyi/ETDataset |
| Exchange | 8 | Daily | Exchange Rate Dataset | https://github.com/laiguokun/multivariate-time-series-data |

#### Download Tasks
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| Download Electricity dataset | ⬜ Pending | - | 321 variables, hourly |
| Download Traffic dataset | ⬜ Pending | - | 862 variables, hourly |
| Download Weather dataset | ⬜ Pending | - | 21 variables, 10-min |
| Download ETTh1 dataset | ⬜ Pending | - | 7 variables, hourly |
| Download ETTh2 dataset | ⬜ Pending | - | 7 variables, hourly |
| Download ETTm1 dataset | ⬜ Pending | - | 7 variables, 15-min |
| Download ETTm2 dataset | ⬜ Pending | - | 7 variables, 15-min |
| Download Exchange dataset | ⬜ Pending | - | 8 variables, daily |
| Create data provider module | ✅ Complete | 2026-05-13 | `data_provider.py` |
| Add dataset references to thesis | ⬜ Pending | - | BibTeX entries |

### Phase 3: Contribution 1 - Fuzzy BPA-EGNN Implementation

#### 3.1 Graph Construction Module
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| TimeSeriesGraphConstructor class | ✅ Complete | 2026-05-13 | `models/graph_constructor.py` |
| Temporal adjacency computation | ✅ Complete | 2026-05-13 | Fuzzy membership-based |
| Variable adjacency computation | ✅ Complete | 2026-05-13 | Correlation-based |
| Adaptive adjacency matrix | ✅ Complete | 2026-05-13 | Learnable parameters |
| DynamicGraphConstructor class | ✅ Complete | 2026-05-13 | Attention-based |
| Graph normalization | ✅ Complete | 2026-05-13 | Symmetric normalization |

#### 3.2 Fuzzy BPA Module
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| FuzzyMembershipFunction class | ✅ Complete | 2026-05-13 | `models/fuzzy_bpa.py` |
| Gaussian membership | ✅ Complete | 2026-05-13 | μ(x) = exp(-(x-c)²/2σ²) |
| Triangular membership | ✅ Complete | 2026-05-13 | μ(x) = max(0, 1 - \|x-c\|/w) |
| Trapezoidal membership | ✅ Complete | 2026-05-13 | Four-parameter function |
| Sigmoid membership | ✅ Complete | 2026-05-13 | μ(x) = 1/(1 + exp(-k(x-c))) |
| EvidenceMachineKernel class | ✅ Complete | 2026-05-13 | Maps features to evidence |
| FuzzyBPAModule class | ✅ Complete | 2026-05-13 | Main BPA computation |
| Dempster's combination rule | ✅ Complete | 2026-05-13 | Classic combination |
| Murphy's combination rule | ✅ Complete | 2026-05-13 | Average combination |
| Yager's combination rule | ✅ Complete | 2026-05-13 | Conflict to universal set |
| BeliefFunction class | ✅ Complete | 2026-05-13 | Belief/Plausibility computation |

#### 3.3 EGNN Integration
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| NodeUpdateNetwork class | ✅ Complete | 2026-05-13 | `models/egnn_layer.py` |
| EdgeUpdateNetwork class | ✅ Complete | 2026-05-13 | Similarity computation |
| EGNNLayer class | ✅ Complete | 2026-05-13 | Combined layer |
| MultiLayerEGNN class | ✅ Complete | 2026-05-13 | Deep EGNN |
| GraphAttentionLayer class | ✅ Complete | 2026-05-13 | Alternative attention |

#### 3.4 Main Model
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| FuzzyBPAEGNN class | ✅ Complete | 2026-05-13 | `models/fuzzy_bpa_egnn.py` |
| FuzzyBPAEGNNConfig class | ✅ Complete | 2026-05-13 | Configuration |
| Normalization layer | ✅ Complete | 2026-05-13 | Input normalization |
| Temporal projection | ✅ Complete | 2026-05-13 | seq_len to pred_len |
| Evidence fusion | ✅ Complete | 2026-05-13 | Multiple methods |
| Uncertainty quantification | ✅ Complete | 2026-05-13 | predict_with_uncertainty() |

#### 3.5 Training Pipeline
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| TimeSeriesDataset class | ✅ Complete | 2026-05-13 | `train.py` |
| Trainer class | ✅ Complete | 2026-05-13 | Main training loop |
| Multi-GPU support (DDP) | ✅ Complete | 2026-05-13 | DistributedDataParallel |
| Mixed precision training | ✅ Complete | 2026-05-13 | AMP support |
| Learning rate scheduling | ✅ Complete | 2026-05-13 | Cosine, Step, Plateau |
| Early stopping | ✅ Complete | 2026-05-13 | Patience-based |
| Model checkpointing | ✅ Complete | 2026-05-13 | Save/Load |
| Weights & Biases integration | ✅ Complete | 2026-05-13 | Experiment tracking |

### Phase 4: Contribution 2 - Alternative Evidence Modules & Explainability

#### 4.1 Transferable Belief Model (TBM)
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| TransferableBeliefModel class | ✅ Complete | 2026-05-13 | `models/evidence_modules.py` |
| Conjunctive combination | ✅ Complete | 2026-05-13 | No normalization |
| Disjunctive combination | ✅ Complete | 2026-05-13 | Union-based |
| Pignistic transformation | ✅ Complete | 2026-05-13 | Decision making |
| Open world assumption | ✅ Complete | 2026-05-13 | Empty set mass |
| Conflict computation | ✅ Complete | 2026-05-13 | K value |

#### 4.2 Pignistic Transformation Module
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| PignisticTransformationModule class | ✅ Complete | 2026-05-13 | `models/evidence_modules.py` |
| Forward transformation | ✅ Complete | 2026-05-13 | BPA to probability |
| Inverse transformation | ✅ Complete | 2026-05-13 | Probability to BPA |
| Pignistic distance | ✅ Complete | 2026-05-13 | L2 distance |
| Decision making | ✅ Complete | 2026-05-13 | Max belief/probability |

#### 4.3 Evidential k-NN Module
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| EvidentialKNNModule class | ✅ Complete | 2026-05-13 | `models/evidence_modules.py` |
| Distance computation | ✅ Complete | 2026-05-13 | Euclidean |
| Distance to mass conversion | ✅ Complete | 2026-05-13 | Exponential decay |
| Dempster combination | ✅ Complete | 2026-05-13 | Multi-source fusion |
| Learnable distance weight | ✅ Complete | 2026-05-13 | Parameter |

#### 4.4 Credal Classification Module
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| CredalClassificationModule class | ✅ Complete | 2026-05-13 | `models/evidence_modules.py` |
| Credal set computation | ✅ Complete | 2026-05-13 | [Bel, Pl] intervals |
| Precise classification | ✅ Complete | 2026-05-13 | Single class |
| Imprecise classification | ✅ Complete | 2026-05-13 | Multiple classes |
| Rejection option | ✅ Complete | 2026-05-13 | High uncertainty |
| Credal distance | ✅ Complete | 2026-05-13 | Hausdorff |

#### 4.5 Explainability Tool
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| BeliefVisualizer class | ✅ Complete | 2026-05-13 | `explainability.py` |
| BPA visualization | ✅ Complete | 2026-05-13 | Bar charts |
| Belief/Plausibility plots | ✅ Complete | 2026-05-13 | Interval plots |
| Uncertainty distribution | ✅ Complete | 2026-05-13 | Histograms |
| Evidence combination plots | ✅ Complete | 2026-05-13 | Multi-source |
| UncertaintyMetrics class | ✅ Complete | 2026-05-13 | Metrics computation |
| Total uncertainty | ✅ Complete | 2026-05-13 | Pl - Bel |
| Non-specificity | ✅ Complete | 2026-05-13 | Yager's measure |
| Conflict measure | ✅ Complete | 2026-05-13 | Entropy-based |
| Ambiguity measure | ✅ Complete | 2026-05-13 | 1 - max(belief) |
| SHAPExplainer class | ✅ Complete | 2026-05-13 | Feature importance |
| EvidenceDecomposer class | ✅ Complete | 2026-05-13 | Contribution analysis |
| Temporal contribution | ✅ Complete | 2026-05-13 | Time step importance |
| Variable contribution | ✅ Complete | 2026-05-13 | Feature importance |
| ModelExplainer class | ✅ Complete | 2026-05-13 | Unified interface |

#### 4.6 Factory Pattern
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| EvidenceModuleFactory class | ✅ Complete | 2026-05-13 | Module creation |

### Phase 5: Experiments

#### 5.1 Experiment Infrastructure
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| run_experiments.py | ✅ Complete | 2026-05-13 | Main experiment runner |
| run_experiments.sh | ✅ Complete | 2026-05-13 | Batch script |
| Dataset configurations | ✅ Complete | 2026-05-13 | All datasets |
| Prediction horizons | ✅ Complete | 2026-05-13 | 96, 192, 336 |
| Evidence method configs | ✅ Complete | 2026-05-13 | Dempster, Murphy, Yager, Average |

#### 5.2 Run Experiments
| Dataset | Pred Len | Dempster | Murphy | Yager | Average | Status |
|---------|----------|----------|--------|-------|---------|--------|
| ETTh1 | 96 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| ETTh1 | 192 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| ETTh1 | 336 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| ETTh2 | 96 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| ETTh2 | 192 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| ETTh2 | 336 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Weather | 96 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Weather | 192 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Weather | 336 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Exchange | 96 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Exchange | 192 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Exchange | 336 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Electricity | 96 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Electricity | 192 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Electricity | 336 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Traffic | 96 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Traffic | 192 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |
| Traffic | 336 | ⬜ | ⬜ | ⬜ | ⬜ | Pending |

#### 5.3 Results Generation
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| Generate comparison table | ⬜ Pending | - | LaTeX format |
| Generate plots | ⬜ Pending | - | MSE/MAE comparison |
| Statistical analysis | ⬜ Pending | - | Significance tests |

### Phase 6: Thesis Updates

#### 6.1 Introduction Updates
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| Add Contribution 2 context | ⬜ Pending | - | Section 1.1 |
| Update objectives | ⬜ Pending | - | Section 1.2 |
| Update research questions | ⬜ Pending | - | Section 1.3 |
| Update contributions list | ⬜ Pending | - | Section 1.4 |

#### 6.2 Literature Review Updates
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| Add TBM section | ⬜ Pending | - | Smets' work |
| Add Pignistic transformation section | ⬜ Pending | - | Decision theory |
| Add Evidential k-NN section | ⬜ Pending | - | Denoeux's work |
| Add Credal classification section | ⬜ Pending | - | Imprecise probability |
| Add SHAP explainability section | ⬜ Pending | - | Lundberg & Lee |
| Add uncertainty quantification section | ⬜ Pending | - | Epistemic uncertainty |
| Add references | ⬜ Pending | - | BibTeX entries |

#### 6.3 Chapter 3 (Contribution 1) Updates
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| Write motivation | ⬜ Pending | - | Why Fuzzy BPA + EGNN |
| Write methodology | ⬜ Pending | - | Technical details |
| Write experimental setup | ⬜ Pending | - | Datasets, baselines |
| Add simulation results | ⬜ Pending | - | Tables, figures |
| Write summary | ⬜ Pending | - | Key findings |

#### 6.4 Chapter 4 (Contribution 2) Creation
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| Create chapter file | ⬜ Pending | - | Chap04/ |
| Write motivation | ⬜ Pending | - | Why alternatives |
| Write TBM methodology | ⬜ Pending | - | Section 4.2.1 |
| Write Pignistic methodology | ⬜ Pending | - | Section 4.2.2 |
| Write EKNN methodology | ⬜ Pending | - | Section 4.2.3 |
| Write Credal methodology | ⬜ Pending | - | Section 4.2.4 |
| Write explainability methodology | ⬜ Pending | - | Section 4.3 |
| Write experimental setup | ⬜ Pending | - | Section 4.4 |
| Add simulation results | ⬜ Pending | - | Section 4.5 |
| Write summary | ⬜ Pending | - | Section 4.6 |

#### 6.5 Conclusion Updates
| Task | Status | Date Completed | Notes |
|------|--------|----------------|-------|
| Summarize Contribution 1 | ⬜ Pending | - | Key findings |
| Summarize Contribution 2 | ⬜ Pending | - | Comparative analysis |
| Future work | ⬜ Pending | - | Extensions |

---

## File Structure

```
k_zoodip/egnn/fuzzy_bpa_egnn/
├── models/
│   ├── __init__.py           ✅ Complete
│   ├── graph_constructor.py  ✅ Complete
│   ├── fuzzy_bpa.py          ✅ Complete
│   ├── egnn_layer.py         ✅ Complete
│   ├── fuzzy_bpa_egnn.py     ✅ Complete
│   └── evidence_modules.py   ✅ Complete
├── train.py                  ✅ Complete
├── data_provider.py          ✅ Complete
├── explainability.py         ✅ Complete
├── run_experiments.py        ✅ Complete
├── run_experiments.sh        ✅ Complete
├── requirements.txt          ✅ Complete
├── README.md                 ✅ Complete
└── PLAN_PROGRESS.md          ✅ Complete (this file)
```

---

## Summary Statistics

| Category | Total | Complete | Pending | Progress |
|----------|-------|----------|---------|----------|
| Project Setup | 4 | 4 | 0 | 100% |
| Datasets | 10 | 2 | 8 | 20% |
| Contribution 1 | 26 | 26 | 0 | 100% |
| Contribution 2 | 28 | 28 | 0 | 100% |
| Experiments | 22 | 4 | 18 | 18% |
| Thesis Updates | 23 | 5 | 18 | 22% |
| **Total** | **113** | **69** | **44** | **61%** |

---

## Recently Completed Tasks

### 2026-05-13
- ✅ Created dataset download script (`download_datasets.py`)
- ✅ Created Chapter 3 template (Contribution 1 - Fuzzy BPA-EGNN)
- ✅ Created Chapter 4 template (Contribution 2 - Alternative Evidence Modules)
- ✅ Created references.bib with all BibTeX entries
- ✅ Added dataset sources and links to plan
- ✅ Created Chapter 1 Introduction updates template
- ✅ Created Chapter 2 Literature Review updates template

---

## File Structure (Updated)

```
k_zoodip/egnn/fuzzy_bpa_egnn/
├── models/
│   ├── __init__.py           ✅ Complete
│   ├── graph_constructor.py  ✅ Complete
│   ├── fuzzy_bpa.py          ✅ Complete
│   ├── egnn_layer.py         ✅ Complete
│   ├── fuzzy_bpa_egnn.py     ✅ Complete
│   └── evidence_modules.py   ✅ Complete
├── thesis/
│   ├── Chapter1_Introduction_Updates.tex  ✅ Complete
│   ├── Chapter2_LiteratureReview_Updates.tex  ✅ Complete
│   ├── Chapter3_Contribution1.tex        ✅ Complete
│   ├── Chapter4_Contribution2.tex        ✅ Complete
│   └── references.bib                    ✅ Complete
├── train.py                  ✅ Complete
├── data_provider.py          ✅ Complete
├── explainability.py         ✅ Complete
├── run_experiments.py        ✅ Complete
├── run_experiments.sh        ✅ Complete
├── download_datasets.py      ✅ Complete
├── requirements.txt          ✅ Complete
├── README.md                 ✅ Complete
└── PLAN_PROGRESS.md          ✅ Complete (this file)
```

---

## Next Steps

1. **Immediate Priority:**
   - Download datasets (Electricity, Traffic, Weather, ETT, Exchange)
   - Run experiments on all datasets and prediction horizons
   - Generate comparison results

2. **Thesis Writing:**
   - Update Introduction with Contribution 2 context
   - Add new sections to Literature Review
   - Create Chapter 4 for Contribution 2
   - Update all results in thesis

---

## Notes

- All code implementation is complete for both contributions
- The model is ready for training and evaluation
- Datasets need to be downloaded from the Time Series Library
- Thesis writing tasks remain pending

---

*This file should be updated as progress is made on each task.*
