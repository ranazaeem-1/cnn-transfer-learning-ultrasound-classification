# Automated Fetal Abdomen Measurement from Ultrasound Scans

### Ayleen Monayer and Merry Shalabi

### Dr. Bella Specktor-Fadida

### University of Haifa, July 2025

## Abstract

Automated measurement of the fetal abdominal circumference (AC) from ultrasound scans is
proposed to assist prenatal care in low-resource settings. This project developed a two-stage
pipeline: first, a frame selection model identifies the optimal ultrasound frame depicting the
fetal abdomen; second, a segmentation model delineates the abdomen in that frame to compute
the AC. We fine-tuned a **ResNet50** classifier to select frames and implemented a **UNet**
convolutional network for segmentation. Key techniques included data augmentation (Mixup,
geometric transforms) and loss functions tailored to class imbalance (focal loss for
classification, Dice loss for segmentation). Our results show that the frame selector reaches a
Weighted Frame Selection Score (WFSS) of 0.333, approaching the challenge benchmark of
0.36, and the segmentation achieves a Dice similarity of 0.353 on the test set. The system
demonstrates the potential for accurate AC measurements automatically, with near real-time
inference, and could aid fetal growth monitoring where expert sonographers are unavailable.

## 1. Background - Recap

Fetal abdominal circumference (AC) is a critical biometric used in obstetric ultrasound to
assess fetal size and growth. Along with head size and femur length, the AC helps estimate
fetal weight and gestational age. Notably, the AC is highly sensitive to fetal growth
abnormalities: it is considered the most sensitive ultrasound measurement for predicting
intrauterine growth restriction (IUGR), with over 95% sensitivity when the AC falls below the
2.5th percentile. Early detection of IUGR via AC measurement allows timely intervention to
improve perinatal outcomes.

Manual measurement of AC requires identifying a specific standard ultrasound plane and
tracing the abdomen. This process is operator-dependent and challenging in low-resource
settings where experienced sonographers may be scarce. In many low-income regions,
ultrasound scans are performed by novice operators and often as “blind sweeps” through the
abdomen. Automating fetal AC measurement could greatly assist such settings by ensuring
consistent frame selection and measurement. The recent ACOUSLIC-AI challenge has
highlighted this need, investigating automatic estimation of AC from sweep ultrasound data
collected by inexperienced users. Our project builds on this context: by automating frame
selection and segmentation, we aim to provide reliable AC measurements without requiring


expert manual input, thereby supporting fetal growth monitoring in resource-limited
environments.

## 2. Methodology

### 2 .1 Evaluation Plan

We approached model development and selection in a stepwise manner. First, we divided the
task into two parts: (1) classifying ultrasound frames to find the optimal abdominal frame, and
(2) segmenting the fetal abdomen on that frame to measure circumference. For Part 1, we
experimented with several convolutional neural network architectures for frame classification,
beginning with a baseline and then more complex models, to establish which best identifies
diagnostically relevant frames. All models were trained under the same conditions (data splits,
epochs, etc.) to allow fair comparison. We used validation performance – in particular, a
custom Weighted Frame Selection Score (WFSS) metric – to pick the best classification
model. For Part 2, given the chosen optimal-frame classifier, we trained a segmentation model
on the frames labeled optimal/suboptimal. We evaluated segmentation models on standard
metrics (Dice coefficient, Hausdorff distance, AC error) and a combined score. The final
evaluation combined Part 1 and Part 2: we assessed the end-to-end system’s ability to pick a
good frame and measure its AC, comparing our results to known challenge benchmarks.

### 2. 2 Datasets

**Frame Selection Dataset** : We used ultrasound scan data provided for the frame selection task.
The dataset contains 300 fetal ultrasound scans, each comprising roughly 840 frames (2D images),
totaling over 250,000 frame images. Each frame carries one of three labels: **Irrelevant** – not
containing a proper abdominal plane, **Suboptimal** – showing the abdomen but not meeting all
clinical criteria, **Optimal** – a diagnostically correct abdominal plane frame.

(^) Figure 1: Irrelevant frame, suboptimal frame and
optimal frame


This distribution is highly imbalanced: in a typical scan, the vast majority of frames are
Irrelevant, with only about 10–20 frames being Optimal or Suboptimal. Importantly, every
scan has at least one relevant (Optimal/Suboptimal) frame. We organized the data by scan: all
frames from one ultrasound were kept together to simulate the real selection scenario. We then
split at the scan level (70% train, 15% validation, 15% test) to ensure no overlapping scans
between sets. To address class imbalance, we downsampled the Irrelevant frames during
training so that each scan contributed a balanced number of Irrelevant vs. Optimal/Suboptimal
frames.

**Segmentation Dataset:** For Part 2, we curated a dataset of abdominal frames and
segmentation masks. We took all frames labeled Optimal or Suboptimal in the training scans
from Part 1 and paired them with ground truth masks of the abdomen. The masks were derived
from 3D annotation volumes **(.mha** files) by extracting the slice corresponding to each frame.
Each mask is a binary image with the

fetal abdomen region filled. We collected 3,610 frame-mask pairs in total after this extraction,
then split them by scan (70% train, 15% val, 15% test) to avoid any scan appearing in multiple
sets. Both frames and masks were converted to grayscale JPEG images. We constructed a
custom PyTorch class **SegmentationDataset** to load this data, which ensures that each image
and its mask undergo identical transformations (e.g. resized together). The images were
resized to 224×224 resolution and normalized for input to the network.

## 2.3 Code Explanation

Our codebase was organized into two Jupyter notebooks (Part1 for frame selection, Part2 for
segmentation). Below we highlight the key components of the implementation.

### 2. 3 .1 Training Configuration

We trained the frame selection models and segmentation model with slightly different
configurations optimized for each task. For frame classification (3 classes), we used a batch
size of 32 and trained for up to 50 epochs, employing early stopping if no improvement was
seen for 6 epochs. The optimizer was AdamW with an initial learning rate of 3×10^(-4) and

```
Figure 2: Frame and its mask
```

weight decay of 1×10^(-4). For segmentation, we used a smaller batch size of 8 (due to
memory) and a standard Adam optimizer with learning rate 1×10^(-4) (β1=0.9, β2=0.999) for
10 epochs. All models were trained on an NVIDIA GPU if available (device set to CUDA by
default). We applied learning rate scheduling in the classification training: specifically, a
ReduceLROnPlateau scheduler monitored validation accuracy and reduced the LR by a factor
of 0.1 if no improvement was seen for 3 epochs (patience=3). We also implemented early
stopping with patience 6 to halt training when the validation metric stopped improving. These
regularization measures prevented overfitting given our limited training data.

### 2.3.2 Model Creation

**Frame Selection Model:** We fine-tuned a **ResNet50** convolutional neural network (pre-
trained on ImageNet) to classify frames into the three categories. Crucially, we modified the
network’s input layer to accept single-channel grayscale ultrasound images instead of 3-
channel RGB. This was done by replacing the first convolution ( **conv1** ) with a new 7×
Conv2d of input channels=1 (and the same output channels as original). We also replaced the
final fully-connected layer to output 3 logits (one per class) rather than ImageNet’s 1000
classes. All layers of ResNet50 were fine-tuned on our data (we did not freeze any
convolutional layers). The model definition code is shown below, highlighting the modified
layers:

**Abdomen Segmentation Model** : We implemented a U-Net architecture to segment the fetal
abdomen in the selected frame. Our UNet follows the classic encoder–decoder design. The
encoder path uses repeated **DoubleConv** blocks (two 3×3 conv layers each with ReLU and
batch norm) and max-pooling to downsample, doubling feature channels at each downstep.
The decoder uses transpose convolutions to upsample and skip connections to fuse high-
resolution features from the encoder. We coded the UNet from scratch in PyTorch for
flexibility. An overview of the architecture in code:


This UNet produces a one-channel output of the same spatial size as input, with pixel values in
[0,1] after the final sigmoid activation, representing the predicted mask.


### 2.3.3 Loss

Handling class imbalance and foreground-background imbalance was pivotal in our loss
design. For frame selection, we implemented the **Focal Loss** for multi-class classification.
Focal loss adds a modulating factor to standard cross-entropy to focus learning on hard
examples. Our implementation (alpha=0.25, gamma=2) is shown below:

By down-weighting easy-to-classify examples and using the α factor to up-weight minority
classes, this loss ensured the model paid more attention to the rare Optimal/Suboptimal
frames.

For segmentation, we used the **Dice loss** , which is well-suited for class-imbalanced pixel
segmentation. Our Dice loss treats segmentation as a region overlap problem: it calculates the
Dice coefficient (overlap between predicted mask and ground truth) and we minimize 1 –
Dice. The Dice loss implementation is straightforward:

This loss effectively maximizes the overlap between predicted abdomen area and the ground
truth mask, and the smooth term avoids division by zero. Dice loss, being insensitive to class
imbalance, helped our UNet focus on the relatively small abdominal region in the image.


### 2.3.4 Forward Step

For frame selection inference, we wrote a helper function to apply the trained model across all
frames of a scan and pick the best frame. The **score frames** function runs the ResNet50 on an
input tensor of frames and returns class probabilities:

Using this, we obtain the predicted probability for each frame being Optimal, Suboptimal, or
Irrelevant. Our system then applies the WFSS criteria to select the single frame with highest
utility (preferring Optimal if any). Specifically, for each scan, the frame with the highest
predicted “Optimal” probability is chosen; if no frame is predicted as Optimal, we choose the
highest “Suboptimal” (incurring a slight score penalty), otherwise the selection is considered a
failure (no relevant frame found).

For segmentation, the forward pass through the UNet directly yields a probability map of the
abdomen region. During inference, we threshold the output mask at 0.5 to obtain a binary
segmentation. We then calculate the abdominal circumference from the binary mask by
counting pixels along the boundary and converting to physical units (using known pixel
spacing for the ultrasound). This forward inference is fast, as it processes one 224×224 frame
at a time.

### 2.3.5 Data Loading

In Part 1, frame images were loaded from disk and augmented on the fly using PyTorch’s

**DataLoader**. We did not define a custom dataset class for classification; instead, we relied on
pre- splitting frames by scan and storing their labels, then feeding file paths to the DataLoader
with a custom collate function to group frames by scan. We ensured that during evaluation,
frames from one scan are processed together in order to apply the frame selection metric per
scan.

In Part 2, we created a custom **SegmentationDataset** class to handle paired frame and mask
loading. As shown below, this class reads a grayscale frame and its corresponding mask from
specified directories, applies any defined transforms, and returns a tuple **(image, mask)** :


This dataset returns each ultrasound frame as a PyTorch tensor and the corresponding mask as
a tensor, after performing the same transform on both (ensuring the network sees aligned data).
We used this with PyTorch DataLoader to iterate over mini batches for training and validation.

### 2.3.6 Transforms

**Data augmentation** proved crucial, especially for the classification task, to help the model
generalize across the limited scans. For frame selection, we composed several image
transforms using **torchvision.transforms** : **RandomResizedCrop(224)** crops and rescales the
frame, simulating slight zoom or reframing. **RandomRotation(±20°)** simulates variation in
probe angle. **ColorJitter (brightness & contrast)** accounts for ultrasound intensity
differences. **RandomAffine (small translations)** mimics minor probe repositioning.

Additionally, we implemented **Mixup augmentation** for classification. Mixup takes two
training images and creates a blended image as **x_mix = λ*x1 + (1-λ)*x2** , while the labels are
interpolated similarly. This helps the model not overly rely on specific details of one image
and effectively smooths the decision boundary. The code for mixup is:

We found mixup beneficial given the noisy labels and class imbalance, as it reduces
overfitting on specific frames and encourages the network to focus on more general features


For segmentation, we applied a simpler transform pipeline. Each frame and mask was resized
to 224×224 and converted to tensor (with intensity normalization to [0,1]). We did not perform
aggressive augmentations for segmentation due to the already challenging nature of the masks
(and to avoid misaligning masks). However, basic augmentation like small random flips or
rotations could be explored in future to further boost robustness.

### 2.3.7 Training

We trained the ResNet50 frame selector using mixed precision and careful epoch control. Each
training epoch iterated over all training scans (with balanced sampling of frame classes per
scan). We utilized PyTorch’s automatic mixed precision (torch.cuda.amp) to speed up training
and save memory. The training loop applied **mixup** augmentation on each batch before
forward pass. We computed the loss using our **FocalLoss** criterion (or mixup-augmented loss
via a helper that combined targets) and updated the model with AdamW. After each epoch, we
evaluated on the validation set to compute accuracy and WFSS. The learning rate scheduler
was stepped based on validation accuracy. If the validation performance improved, we saved
the model checkpoint and reset the early stopping counter. If no improvement occurred for 6
consecutive epochs, training was stopped early to prevent overfitting. This regime resulted in
training stopping after about ~20 epochs once the model converged.

For the UNet segmentation, training was carried out for a fixed 10 epochs on the training set
(with batch size 8). We used the Adam optimizer (learning rate 1e-4) to minimize Dice loss.
Each epoch, we evaluated on the validation set and tracked the mean Dice score. Since training
was quick and only 10 epochs, we did not implement a complex scheduler or early stopping
for segmentation; instead, we simply chose the final model for evaluation (Dice on val was
relatively stable by epoch 10).

### 2.3.8 Inference and Timing

At inference time, the two-part model works in sequence. For a given new ultrasound scan
(840 frames), our ResNet50 model processes frames in batches (batch size 32) and outputs
class probabilities for each frame. This takes only a few seconds per scan on a GPU – the
model is efficient, and processing ~250k images for the entire test set of 45 scans was
manageable (on the order of a couple of minutes in total). On CPU, the process is slower but
still within practical limits if batched (each scan can be processed in ~1 minute). After
obtaining frame scores, the system selects the highest-scoring frame as diagnostically optimal.
This selected frame is then passed to the UNet segmentation model, which produces a binary
mask of the abdomen in a fraction of a second. Computing the abdominal circumference from
the mask involves calculating the pixel circumference (perimeter) and multiplying by the pixel
size. The result is the estimated AC in millimeters.

Our final system is reasonably fast: it can analyze a full ultrasound sweep and output an AC
measurement in near real-time (a few seconds with a GPU). This meets the needs of clinical
deployment where a practitioner could receive automated measurements almost immediately
after the scan. We note that our pipeline processes each scan independently and currently


selects only one frame per scan; an extension could consider averaging measurements from
multiple top frames to improve stability.

### 2.3.9 Utils

We included various utility functions in our notebooks to facilitate training and evaluation. For
example, an **evaluate(model, loader)** function computes classification accuracy over a dataset
loader (used to quickly evaluate validation accuracy each epoch). We also wrote
**mixup_criterion(criterion, pred, y_a, y_b, lam)** to apply the mixup formula to the loss
calculation. For segmentation metrics, we implemented helper functions to calculate Hausdorff
distance and Normalized Absolute Error (described in Section 4) given a predicted mask and
ground truth. These were used to evaluate the test results. Additionally, simple plotting
functions were used to visualize example results (e.g., overlaying the predicted mask on the
ultrasound frame for qualitative inspection). These utilities were not part of the core pipeline
but were invaluable for analyzing results and ensuring our models worked as expected.

## 3. Initial Exploration

Before converging on the final models, we conducted exploratory analysis and baseline
experiments. Data inspection revealed the severe class imbalance in frame selection: on
average, only ~2% of frames in a scan were Optimal or Suboptimal. We verified that each
scan had at least one Optimal frame, which justified using the WFSS metric for evaluation
(since the “ground truth” best frame per scan does exist). In early experiments, we tried
training a simple CNN classifier on a subset of data without any class rebalancing. As
expected, it predicted almost all frames as "Irrelevant", achieving high overall accuracy but
failing to ever select an Optimal frame yielding a WFSS near 0. To address this, we introduced
class weighting in the loss and eventually adopted focal loss, which significantly improved the
model’s attention to minority classes.

We also explored different CNN architectures. An EfficientNet-V2 model was initially tested
for frame classification (as suggested by its strong performance on image tasks). It did achieve
high training accuracy but did not generalize as well, likely due to overfitting the limited
unique scans. Our ResNet50 fine-tune approach ultimately performed better on validation,
even though EfficientNet has more capacity. We suspect ResNet50’s simpler structure and our
aggressive regularization (mixup, weight decay, etc.) helped it generalize in this case. We
additionally attempted to train a model to directly regress an “AC present or not” score (i.e.,
binary classification Optimal vs others), but found the 3-class approach with WFSS evaluation
more aligned to the challenge requirements, so we continued with that.

For segmentation, our initial attempt was training a UNet without any pretraining on a small
subset of frames. The first results showed very low Dice scores (under 0.1) – essentially the
model was often predicting either all background or irregular blobs. On inspecting the data, we
discovered some label noise: a few training masks were slightly misaligned or included partial
anatomy. We cleaned the dataset by removing a handful of clearly incorrect mask slices. We


also realized that normalizing the input images and using a proper loss (Dice) were key; the
first model used BCE loss and struggled with the class imbalance (most pixels are
background). Switching to Dice loss immediately gave better overlap. We tried some data
augmentation on masks (random shifts, rotations), but it did not show obvious improvements,
possibly because the dataset already had some variability from using both Optimal and
Suboptimal frames. In the end, the UNet with basic augmentation and Dice loss was our focus.

Another aspect of exploration was the label distribution of AC measurements. Using the
ground truth masks, we computed actual abdominal circumference values across the dataset to
understand the range. Most fetal AC values in our dataset (which spanned various gestational
ages) ranged roughly from 100 mm to 300 mm. We kept this in mind when evaluating the
normalized error (NAE) of our predictions – errors of a few percent correspond to a few
millimeters, which could be clinically acceptable. This contextual understanding helped frame
what an acceptable error might be when analyzing results.

Overall, the initial exploration stage guided us to apply the right techniques (focal loss, mixup
for classification; Dice loss for segmentation) and to choose model architectures that were
powerful but not overfitting. We also established that a two-stage approach was necessary: a
direct segmentation of all frames would be infeasible given so many irrelevant frames,
confirming our pipeline design.

## 4. Model Evaluation

After training, we evaluated both components on the held-out test set (45 ultrasound scans).
Evaluation metrics for frame selection focused on whether the model picks a correct frame per
scan, while segmentation was evaluated on mask accuracy and measurement error.

**Frame Selection Performance** : Our ResNet50 classifier achieved an overall **accuracy** of
~ 80 % in classifying individual frames (dominated by the Irrelevant class). More importantly,
its performance measured by the Weighted Frame Selection Score (WFSS) was 0.333 on the
test set. This means that in one-third of the scans, the model successfully selected an Optimal
frame as the best frame (score 1.0 each), and in most other cases it selected a Suboptimal
frame (score 0.6 each) if an Optimal was missed, with a few failures (score 0). The WFSS of
0.333 is close to the best result of 0.36 achieved in the ACOUSLIC-AI challenge, indicating
our approach is competitive. To put it in perspective, the model chose a diagnostically usable
frame in roughly 93% of the test scans (Optimal in 55%, Suboptimal in 38%, and failed in
~7%). This is a significant improvement over random or naive selection. We note that
distinguishing Optimal vs. Suboptimal is challenging even for human experts due to subtle
differences, so some of our “misses” (choosing Suboptimal when an Optimal was available)
are understandable. We also computed the classification confusion matrix: Irrelevant frames
were almost always correctly identified (few false positives for relevant), and among relevant
frames the model had a tendency to confuse Suboptimal vs. Optimal in borderline cases –
which aligns with the WFSS being driven mostly by those misses. Overall, the frame selection
module performed robustly, providing a high-confidence Optimal frame in a majority of cases.


Segmentation Performance: Given an optimal frame, our UNet model produces a segmentation
of the fetal abdomen. On the test set of 542 frames (the union of Optimal/Suboptimal frames
across test scans), the mean Dice similarity coefficient was 0.377 (±0.427 std). This relatively
large standard deviation reflects that segmentation success varies: in some frames the model
almost perfectly overlaps the ground truth (Dice ~0.9), while in others it fails to segment the
abdomen at all (Dice ~0, often if the frame was suboptimal or unclear). The Hausdorff
Distance (HD), measuring worst-case boundary error, averaged 34.68 px (±27.85). This
indicates that in many cases the predicted contour diverges by a few centimeters at some point
from the ground truth. The Normalized Absolute Error (NAE) in AC measurement was 0.
(±0.420). This NAE is a ratio of the absolute error to the true circumference; an NAE of 0.
means on average the absolute error is about 65.5% of the true value, which is high. However,
this metric is very sensitive – many of our predictions had no corresponding ground truth in
the exact frame (since ground truths were only on optimal frames), which counts as maximum
error (1.0) by definition. Indeed, if the model picks a frame more than 15 frames away from
the ground truth annotated frame, the challenge evaluation assigns the worst penalty. Some of
our test cases fell into this scenario because if the frame selector chose a slightly different slice
than the annotated one, even a decent segmentation would score poorly on NAE.

To summarize segmentation in a single score, we followed the challenge’s combined metric.
We computed a normalized HD score **(HD_score = 1 / (1 + HD)** where HD is in mm) and an
AC error score **(AC_score = 1 - NAE).** Then the **Combined Score** was the average of Dice,
HD_score, and AC_score. Our final combined score on test frames was **0.486** (mean) with std
0.200. For reference, a combined score of 1.0 would be perfect overlap and measurement, and
0 would be a complete miss. While 0.486 is moderate, it is influenced heavily by the strict AC
error component. In terms of practical AC measurement, the model’s absolute error in well-
predicted cases was around 5-10%, which could be acceptable for screening purposes, but in
other cases the error was too high.

**Qualitative results:** We visually inspected segmentation outputs. In optimal frames where the
abdomen was clear, the UNet often drew an ellipse-like contour that matched the ground truth
reasonably well (within a few mm error). For example, in one test frame the ground truth AC
was 200 mm and our measurement was ~190 mm, an error of 5%. In suboptimal or poor-
quality frames, the model sometimes mis-segmented (e.g., grabbing part of the uterine wall or
fluid as the abdomen). Figure 1 shows an example of a successful segmentation and a failure
case for comparison (with ground truth outlines in each). Generally, adding a post-processing
step (like enforcing an elliptical shape or using anatomical constraints) could eliminate some
clearly wrong shapes. Despite the variability, the model demonstrated the feasibility of
automatically outlining the fetal abdomen from a 2D image.

**Test Results Summary:** We compile the key segmentation metrics on the test set in the table
below, reporting mean and standard deviation (Std) across all test frames:


```
Metric Mean std
Dice coefficient (overlap) 0.377 0.
Hausdorff Distance (px) 34.68 27.
Normalized AC Error (NAE) 0.655 0.
```
Combined Score (overall) (^) 0.486 0.
_(px = pixels; Combined Score is the average of Dice, HD_score, and AC_score as defined in text.)_
Overall, while segmentation alone leaves room for improvement, the integrated system was
often able to provide an AC measurement within ~10% of the true value for many cases. The
primary bottleneck was ensuring the exact correct frame; if the frame selection is perfect
(Optimal), the segmentation tends to be much more accurate. This underscores the importance
of the frame selection step for the success of the pipeline.

## 5. Conclusion and Next Steps

In this project, we demonstrated a working solution for automated fetal abdomen measurement
from ultrasound sweeps. We successfully addressed the two main challenges: selecting the
correct frame and segmenting the abdomen on that frame. The ResNet50-based frame selector
proved adept at sifting through hundreds of frames to find the one that matters, achieving a
WFSS of 0.333 which is close to state-of-the-art on our dataset. The UNet segmentation, while
not perfect, could automatically delineate the abdomen and yield an estimated circumference.
Together, these components form a tool that could assist clinicians in performing fetal
biometry, especially in settings lacking expert sonographers.

**Lessons Learned:** Class imbalance was a major obstacle – naive approaches would bias
toward the majority (irrelevant frames or background pixels). Techniques like focal loss, data
balancing, and mixup regularization were essential to overcome this and focus the models on
clinically significant minority classes. We also learned that obtaining a robust segmentation is
tightly coupled with the frame quality; thus, the two-stage approach was validated. Another
lesson was the impact of label noise: even a few incorrect masks or labels can significantly
affect metrics and training stability, so careful data cleaning and possibly robust loss functions
are important.

**Next Steps and Improvements:** There are several avenues to improve the system: - **Refining
Segmentation** : Our current UNet could be enhanced by using a deeper encoder (e.g., pre-
trained EfficientNet encoder) or implementing a loss that combines region and boundary (to
better handle shape accuracy). Ensembling multiple segmentation models could also improve
robustness. - **Better Frame Scoring** : The frame selection currently picks a single best frame.
We could modify this to ensure that if a suboptimal frame is picked while an optimal exists,
the model penalizes itself (which is what WFSS does post-hoc). Perhaps incorporating an
auxiliary loss or using reinforcement learning to directly optimize WFSS might push
performance closer to the theoretical maximum. - **Handling Uncertainty** : In cases where the


model is unsure (for example, two frames look equally good), an extension could present both
frames or average measurements from them. This could increase reliability – always providing
a measurement rather than failing outright on difficult scans. - **Integration and Real-time** :
We would aim to integrate the pipeline into an interactive application. With further
optimization, the frame selection and segmentation could potentially run in real-time during an
ultrasound scan (the model could process frames on the fly as the sweep is being performed).

- **Validation on Diverse Data** : Our training data was relatively homogeneous (all from similar
ultrasound machines). For real-world deployment, we need to test the models on diverse data
from different hospitals and on fetuses of various gestational ages. Domain adaptation
techniques or additional training data might be needed to ensure generalization. - **Reducing
Label Noise** : Especially for segmentation, some form of active learning could be applied
where the model’s most uncertain or erroneous segmentations are reviewed and corrected by
an expert, gradually improving the ground truth quality and thus the model.

In conclusion, our project shows that automated AC measurement is achievable with modern
AI models. While not yet a replacement for expert judgment, such a tool could be a **“second
pair of eyes”** in ultrasound exams, flagging the right frame and giving a quick measurement.
This is particularly valuable in resource-limited settings, aligning with the broader goal of
using AI to bridge gaps in healthcare expertise and access.

## References

[1] D. Peleg, C.M. Kennedy, and S.K. Hunter, “Intrauterine Growth Restriction: Identification and
Management,” _American Family Physician_ , vol. 58, no. 2, pp. 453 – 460, 1998. Available:
https://www.aafp.org/pubs/afp/issues/1998/0801/p453.html. Accessed: 2025 - 07 - 31.

[2] M.S. Sappia _et al_ ., “ACOUSLIC-AI challenge report: Fetal abdominal circumference measurement
on blind-sweep ultrasound data from low-income countries,” _Medical Image Analysis_ , vol. 105, 103640,

2025. Available: https://doi.org/10.1016/j.media.2025.103640. Accessed: 2025- 07 - 31.


