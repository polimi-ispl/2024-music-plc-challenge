# IEEE-IS² 2024 Music Packet Loss Concealment Challenge
📝 This is the official repository for the [IEEE-IS² 2024 Music Packet Loss Concealment Challenge](https://internetofsounds.net/ieee-is%C2%B2-2024-music-packet-loss-concealment-challenge/). 
Here, partecipants will find the pretrained baseline model and inference code **(to be released on May 20, 2024)**.

## Introduction
**IEEE-IS² 2024 Music Packet Loss Concealment Challenge** is intended to promote research on Packet Loss Concealment (PLC) for Networked Music Performance (NMP) applications.

Packet loss, either by missing packets or high packet jitter, is one of the main problems and, in turn, engineering challenges in real-life NMP. While PLC for Voice over IP has recently attracted a great deal of attention, PLC for NMP applications has been considerably less studied. 

We invite researchers and practitioners in signal processing, machine learning, and audio technologies at large to participate in the first edition of the IEEE-IS² 2024 Music Packet Loss Concealment Challenge.

IEEE-IS² 2024 Music Packet Loss Concealment Challenge is part of the **2nd IEEE International Workshop on Networked Immersive Audio ([IEEE IWNIA 2024](https://internetofsounds.net/2nd-international-workshop-on-networked-immersive-audio/))**, a satellite event of the **5th IEEE International Symposium on the Internet of Sounds ([IEEE IS² 2024](https://internetofsounds.net/is2_2024/))**. The Symposium will be hosted at the International Audio Laboratories Erlangen, Germany, a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS, between Sept. 30 and Oct. 2, 2024.

## Important Dates
-	**May 13, 2024** – Challenge start
-	**May 20, 2024** – Release of the baseline system
-	**June 16, 2024** – Challenge registration deadline
-	**July 3, 2024** – Release of blind test set
-	**July 10, 2024** – Challenge submissions due
-	**Sept 10, 2024** – (tentative) Notification of evaluation results
-	**Sept 30, 2024 - Oct 2, 2024** – Symposium dates
  
## Problem Formulation
Packet-switched networks often rely on best-effort protocols that prioritize speed over reliability. This means there are no guarantees that audio data packets will be correctly transmitted. In fact, excessive delays due to high packet jitter or missing packet may cause gaps in the audio stream at the receiver end.

In NMP applications, the packet size is typically determined by the buffer of the soundcard from which audio data is read. Whereas small buffer sizes are sometimes preferred, this year’s Challenge addresses the more challenging scenario of the packet size being **512 samples**, corresponding to approximately 11.6 ms at a sampling rate of 44.1 kHz. 

All PLC methods submitted to the IEEE-IS² 2024 Music Packet Loss Concealment Challenge are expected to be able to process real-world music signals at a sample rate of 44.1KHz and a bit-depth of 16 bits, i.e., the standard audio specification for Compact Disc Digital Audio.

Moreover, this year’s Challenge will focus on **causal systems only**. 

Namely, at any given time, only previously received packets or prediction thereof may be used to predict the next audio frame. In other words, **systems are not allowed any look ahead**. Please notice that this is different from existing audio deep PLC challenges.

Therefore, **non-causal approaches will not be considered eligible for the Challenge** and will be thus excluded from the final ranking. Other than that, there are no limitations on the proposed PLC methods, which may comprise one or more deep-learning models, traditional signal processing algorithms, or a hybrid approach.

## Registration Procedure
To enter the Challenge, please register on the [EasyChair portal](https://easychair.org/my/conference?conf=is22024paper) by the Challenge registration deadline (see **Important Dates**).
On EasyChair, 
-	Select “make a new submission” in the Author Console.
-	Select “Music Packet Loss Concealment Challenge” as Track.
-	Enter title, abstract, and register all team members as “Authors.”
-	Upload (temporary) placeholder files and complete the submission.

The final files (see **Submission Rules**) can be uploaded by updating an existing submission at any time before the Challenge submission deadline (see **Important Dates**).

**Every team is allowed to submit up to two PLC systems for evaluation.** Each system should be submitted independently of the other by creating a new submission (with a unique title).

## Submission Rules
Once the blind test set is released, every team is required to:
-	Download the blind test set.
-	Process each and every degraded audio clip using the proposed PLC method.
-	Save the enhanced audio clips as 16bit-44.1kHz single-channel wav files making sure to use the same filenames of the corresponding degraded audio clips. 
-	Compress the enhanced audio files in a single zip file.
-	Upload the resulting zip file on a cloud storage service.
-	Make sure that the zip file can be downloaded without access restrictions.

Then, each team should submit the following files through [EasyChair](https://easychair.org/my/conference?conf=is22024paper):
- A 2-page technical report in PDF format detailing their approach `[Upload as: Paper]`
- A txt file containing: `[Upload as: Supplemental material]`
  - a permanent link to a public repository (e.g., GitHub) hosting the project code.
  -	a permanent link to a cloud storage service (e.g., Dropbox) hosting the enhanced audio clips processed with the proposed PLC system. Please make sure that download is enabled without access restrictions.

**The cloud storage folder must host a single zip file containing the enhanced audio files. Make sure that you zip as many audio clips as there are in the blind test set. Every enhanced clip should have the same filename as the corresponding degraded clip. Do not upload the degraded audio clips from the blind test set or any other file to the cloud storage folder.**

Only teams that upload all the required material will be considered eligible for participating in the evaluation process.

**Important: please make sure that the Supplemental material contains both the link to the enhanced audio clips and the link to the project repo.**

### Technical Reports
Technical reports should not exceed 2 pages of scientific content (including tables and figures) + 1 optional page for references only. 

Technical reports should describe the proposed approach and provide all the details to ensure reproducibility. Authors may choose to report additional objective/subjective metrics in their paper. At the end of the Challenge, technical reports will be made available on the [IS² website](https://internetofsounds.net/is2_2024/).

Please notice that **technical reports will <ins>not</ins> be included in the Symposium Proceedings**. Nevertheless, authors are warmly encouraged to submit a regular IS² 2024 paper outlining their PLC method. Regular papers will undergo the normal peer-review process, whereas technical reports will not. 

**Paper acceptance and registration to the conference are not required for taking part in the 2024 Music Packet Loss Concealment Challenge.**

### Official Templates
For drafting the 2-page technical report, please use the same LaTeX and Word template as of regular IS² papers. Official templates can be found here: [IEEE conference templates](https://www.ieee.org/conferences/publishing/templates.html)  

## Evaluation Rules
### Blind Test Set
The blind test set consists of several single-channel audio clips in a 16bit-44.1kHz wav format. Every file consists of a short clip of a closed-miked solo-instrument performance. The dataset will comprise various acoustic instruments, including strings, brass instruments, bass, and classical guitar. 

The audio clips are artificially degraded by dropping packets according to predetermined “packet traces,” i.e., text files containing a string of binary digits: 0 if a packet was correctly received and 1 if the packet was lost. Every digit in a packet trace corresponds to 512 samples. Traces do not contain explicit temporal information, and the packet rate is implicitly determined by the audio sampling rate.

The packet traces used to create the blind test set were repurposed from the INTERSPEECH 2022 Audio Deep Packet Loss Concealment Challenge ([GitHub](https://github.com/microsoft/PLC-Challenge)). 

Said traces are divided into three subsets according to the maximum burst loss length:
-	**Subset 1.** Bursts of up to 6 consecutive packets. 
-	**Subset 2.** Bursts of 6 to 16 consecutive packets.
-	**Subset 3.** Bursts of 16 to 50 consecutive packets.

We sample packet traces from Subset 1 (with high probability) and Subset 2 (with low probability). We do not sample traces from Subset 3.

The resulting packet traces will be made available in txt format along with the corresponding degraded audio clips. In practice, such a degradation consists of filling the missing packets with zeros in the waveform domain. 

The blind test set will be released shortly before the Challenge submission deadline.

Participants are forbidden from using the  blind test set to retrain or tune their models. They should not submit results using other PLC methods other than the one they have developed. **Failing to adhere to these rules will lead to disqualification from the Challenge.**

### Evaluation Procedure
Once the blind test set is made available, participants should process every track using their own PLC system, create a single zip file containing all the resulting audio clips, upload the zip file on a cloud storage service, and submit a permanent link through EasyChair. 

Please do not upload audio files or model weights on EasyChair.

**Participants must save each enhanced clip using the same filename of the corresponding degraded audio file. Failing to do so would make it impossible to automatize the evaluation process and may result in the exclusion of the team’s submission from the final ranking.**

**Note that any ex-post manual intervention on the audio files or any ad-hoc post-processing is forbidden. Every clip in the blind test set should be processed using the proposed PLC method only.**

Since no objective metric was definitively proved to correlate well with human perception when it comes to Music PLC, evaluation will be carried out through a listening test involving a curated subset of the audio clips included in the blind test set. 

In particular, the evaluation will involve a modified **MUSHRA** test (MUltiple Stimuli with Hidden Reference and Anchor) defined by ITU-R Recommendation BS.1534-3. 

In the MUSHRA test, the enhanced clips from each team will be assigned a score on a scale of 0 to 100. For each clip, assessors will be tasked to identify the “Hidden Reference,” i.e., the clean untampered signal, and assign it the highest possible score. At the same time, assessors will be presented with an “Anchor,” i.e., the lower bound of the perceptual scale. Here, zero-filled lossy signals from the blind test set will be used as Anchors. The average score across all MUSHRA ratings will determine the final ranking of each PLC system.

If needed, the submitted audio clips will be converted to mono via channel average and saved as 16bit-44.1kHz wav files before the evaluation. Consider that this procedure is automatic and might affect the audio quality of your submission. Please upload audio in the right format in the first place.

**Evaluation results will be (tentatively) published on September 10th, 2024.**

Several objective metrics will also be computed, including but not limited to mean squared error, log-spectral distortion, and PEAQ. Nonetheless, these metrics will not be taken into consideration for determining the team ranking. 

## Training Data
We do not provide training data, nor do we indicate a list of eligible training datasets. 

However, we prescribe that all training data is exclusively taken from publicly-available freely-accessible audio datasets, i.e., proprietary and/or commercial datasets should not be used for training the proposed PLC systems. This rule applies to any type of training, including pretextual pre-training and fine-tuning.

Every team should clearly state which publicly-available datasets were used in training the final model, provide links or references in the 2-page report, and specify the amount/duration of the audio taken from each and every dataset.

You can augment your data in any way that improves the performance of your model. However, metadata and other auxiliary information other than packet loss traces, if present, should be disregarded and should not be passed to the model in any way. 

Also, participants are strictly prohibited from employing the blind test set for any training or fine-tuning purposes.

## Baseline System
We release a baseline system for the IEEE-IS² 2024 Music Packet Loss Concealment Challenge. The system is a modified PARCnet architecture [1] trained on [Medley-solos-DB](https://zenodo.org/records/3464194) [2].

PARCnet comprises a linear predictor and a lightweight feedforward ConvNet. The linear predictor is fitted in real-time within a sliding context window using the autocorrelation method with white noise compensation, while the ConvNet is trained to estimate the residual of the linear autoregressive model. 

Medley-solos-DB contains 21572 isolated audio clips of clarinet, distorted electric guitar, female singer, flute, piano, tenor saxophone, trumpet, and violin. The Medley-solos-DB instrument set has some overlap with the instruments featured in the blind test set, but it is not a one-to-one correspondence. Moreover, the overall recording conditions vary significantly between the two datasets. Note that the baseline system was not validated on blind test data.

> [1] A. I. Mezza, M. Amerena, A. Bernardini and A. Sarti, "Hybrid Packet Loss Concealment for Real-Time Networked Music Applications," in IEEE Open Journal of Signal Processing, vol. 5, pp. 266-273, 2024, doi: 10.1109/OJSP.2023.3343318.

> [2] V. Lostanlen and C.E. Cella, “Deep convolutional networks on the pitch spiral for musical instrument recognition,” in Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2016.

## Real-Time Performance
Practical PLC systems should take less than a stride length (in ms) to process the next frame. This means that, overall, submitted methods should take less than 11.6 ms to estimate the next 512-sample packet.

However, since we are not collecting and running models during the evaluation phase, we cannot benchmark the respective real-time performance. Therefore, we ask participants to self-assess their own PLC systems by reporting the total number of model parameters and the average time needed to process a single packet on an Intel Core i5 quad-core machine clocked at 2.4 GHz or equivalent processors.

Slower-than-real-time inference is not a reason for disqualifying a team from the Challenge. However, we encourage all participants to respect the real-time constraints as strictly as possible.

## Contacts
For questions, please contact [music.plc.challenge.2024@gmail.com](mailto:music.plc.challenge.2024@gmail.com)

