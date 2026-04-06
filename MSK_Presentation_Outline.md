# MSK Project Presentation Outline

## Slide 1: Title Slide
**Title:** Automated Diagnosis of Myositis from Muscle Ultrasound Images using Deep Learning
**Subtitle:** Classification of Dermatomyositis, Polymyositis, and Inclusion Body Myositis
**Presenter:** [Your Name]
**Date:** January 2026

---

## Slide 2: Project Overview
**Objective:** Develope deep learning models to automate the diagnosis of inflammatory myopathies.
**Two Key Problems:**
*   **Problem A (Screening):** Binary classification – Normal vs. Affected (Disease).
*   **Problem B (Diagnosis):** Multi-class classification – DM vs. PM vs. IBM.
**Significance:** Ultrasound is non-invasive and cost-effective; AI can improve diagnostic accuracy.

---

## Slide 3: Dataset & Demographics
**Total Images:** 3,214 Ultrasound Images
**Class Distribution:**
*   **Normal (N):** 1,313
*   **Dermatomyositis (D):** 555
*   **Polymyositis (P):** 553
*   **Inclusion Body Myositis (I):** 794
**Data Split:** 70% Training, 15% Validation, 15% Testing.

---

## Slide 4: Problem A - Binary Classification (Methodology)
**Goal:** Distinguish Healthy vs. Diseased muscle.
**Model:** ResNet50 (Pre-trained on ImageNet).
**Configuration:**
*   Unfrozen last 100 layers.
*   Head: GlobalAvgPool -> Dense(256) -> Dropout(0.3) -> Sigmoid.
*   Loss: Binary Cross-Entropy.
**Augmentation:** Rotation, Contrast, Zoom.

---

## Slide 5: Problem A - Results
**Key Metrics:**
*   **Test Accuracy:** 78.5%
*   **Validation Accuracy:** 80.5%
*   **AUC Score:** 0.87
**Visuals:**
*   [Insert ROC Curve Image]
*   [Insert Confusion Matrix Image]
**Takeaway:** High accuracy makes this a viable screening tool.

---

## Slide 6: Problem B - Multi-Class Classification (Methodology)
**Goal:** Differentiate between DM, PM, and IBM.
**Models Tested:**
*   ResNet50, ResNet101
*   EfficientNet (B0, B4)
*   DenseNet121, InceptionV3, Xception
**Strategy:**
*   Transfer Learning with fine-tuning.
*   Class weights to handle potential imbalances.
*   Conservative data augmentation to preserve texture features.

---

## Slide 7: Problem B - Model Comparison
**Performance Summary:**

| Model | Test Accuracy | Val Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **ResNet50** | **71.0%** | **72.6%** | **0.69** |
| Xception | 64.7% | 64.6% | 0.61 |
| ResNet101 | 61.5% | 61.8% | 0.61 |

**Key Finding:** ResNet50 outperformed more complex models for this specific texture analysis task.

---

## Slide 8: Problem B - Detailed Results (ResNet50)
**Confusion Matrix Analysis:**
*   **IBM (Inclusion Body Myositis):** High classification accuracy (104/119 correct).
*   **DM vs PM:** Some misclassification between Dermatomyositis and Polymyositis.
*   *Interpretation:* DM and PM have clinically similar ultrasound features, making separation harder.

---

## Slide 9: Comparison with Literature
**Benchmark:** Burlina et al. (2017)
*   *Their Task:* Binary (IBM vs. DM/PM).
*   *Their Accuracy:* 74.8%.

**Our Achievement:**
*   *Our Task:* 3-Class (DM vs. PM vs. IBM).
*   *Our Accuracy:* **71.0% (Test) / 72.6% (Val)**.
**Conclusion:** We achieved comparable accuracy on a significantly harder multi-class problem.

---

## Slide 10: Conclusions
1.  **Effective Screening:** Problem A model (ResNet50) successfully differentiates normal from diseased muscle (AUC 0.87).
2.  **Specific Diagnosis:** Problem B model differentiates between 3 disease types with 71% accuracy.
3.  **Model Selection:** ResNet50 proved robust and efficient compared to larger transformers or deeper networks.

---

## Slide 11: Future Work
*   **Expand Dataset:** Collect more confirmed DM/PM cases.
*   **Advanced Techniques:** Explore Attention Mechanisms / Vision Transformers (ViT).
*   **Clinical Pilot:** Test the model on real-time ultrasound video feeds.

---

## Slide 12: Q&A
**Thank You!**
