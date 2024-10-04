# IEEE-IS² 2024 Music PLC Challenge Baseline

## PARCnet-IS²
PARCnet-IS² is an updated version of PARCnet developed for the [IEEE-IS² 2024 Music PLC Challenge](https://internetofsounds.net/ieee-is%c2%b2-2024-music-packet-loss-concealment-challenge/). 

Compared to the original model, PARCnet-IS² incorporates several minor modifications, primarily due to the higher sampling rate required by the 2024 Challenge (see [Challenge rules](https://github.com/polimi-ispl/2024-music-plc-challenge/blob/main/README.md)). 

If you would like to reference PARCnet-IS² in your work, please cite
```
@article{mezza2024hybrid,
  author={Mezza, Alessandro Ilic and Amerena, Matteo and Bernardini, Alberto and Sarti, Augusto},
  journal={IEEE Open Journal of Signal Processing}, 
  title={Hybrid Packet Loss Concealment for Real-Time Networked Music Applications}, 
  year={2024},
  volume={5},
  number={},
  pages={266-273},
  doi={10.1109/OJSP.2023.3343318}
}
```

---------------

#### Main novelties ✨

- Higher sampling rate: **44.1 kHz**
- Increased packets size: **512** 
- Higher AR model order: **256**

See below for the full list of changes.

---------------

#### Download links 📥

- Download the **model weights** [here](https://polimi365-my.sharepoint.com/:u:/g/personal/10391311_polimi_it/EeMrt4er9jJGg1ksJwoK-gEBBXG6ReI1RCXlwAGAy8m9iw?download=1).

- Download an **exemplary test set** [here](https://polimi365-my.sharepoint.com/:u:/g/personal/10391311_polimi_it/EXHNgqlP101LgUlhp9nB_wgBH20dcxsLRemGBG3TyJCAng?download=1).

## Model description

[PARCnet](https://doi.org/10.1109/OJSP.2023.3343318) (short for *Parallel AutoRegressive model + ConvNet*) was first introduced in 
> A. I. Mezza, M. Amerena, A. Bernardini and A. Sarti, "Hybrid Packet Loss Concealment for Real-Time Networked Music Applications," in IEEE Open Journal of Signal Processing, vol. 5, pp. 266-273, 2024, doi: 10.1109/OJSP.2023.3343318

PARCnet comprises two modules, an autoregressive linear predictor of order $p$ (AR model) and a feed-forward neural network.

The AR model is characterized by

$$ y[n] = \\sum\_{i=1}^{p} \varphi\_i y[n-i] + \varepsilon[n] $$

Having estimated $\varphi_1, ..., \varphi_p$ from a valid context, future samples can be predicted as a linear combination 
of past samples by setting the residual term $\varepsilon[n]$ to zero.

The AR model is fit in real-time from a valid context of 8 past packets (or a prediction thereof) with a stride of 512 samples. We apply the well-known Levinson-Durbin 
algorithm with white noise compensation (a.k.a. diagonal loading of the zeroth autocorrelation coefficient)
to improve the condition number of nearly singular autocorrelation matrices.

In parallel, PARCnet uses a non-autoregressive neural network $f\_\theta$ to predict $\varepsilon[n]$ from a valid context of past samples $\mathbf{x}$ in a single forward pass. 
This yields

$$
\begin{cases}
  \\hat{y}[n] = y[n] + f\_\theta(\mathbf{x})\_{n-k}, \\ 
   y[n] = \Sigma_{i=1}^p \varphi\_i y[n-i].    
\end{cases}
$$

The neural network is a lightweight fully-convolutional feed-forward information bottleneck model 
featuring a classic encoder-decoder structure with 416 K trainable parameters. It is fed a sequence of 8 valid
packets (4096 samples) followed by 768 zeros, corresponding to one 512-sample packet and 256 extra samples 
used to apply a cross-fade between the predicted signal and the subsequent packet, in an attempt to smooth out potential jump 
discontinuities.

In other words, the network is trained to simultaneously predict the residual associated to the valid context and 
the missing packet. At inference time, however, only the last 768 samples are used, whereas the preceding 4096 samples are discarded.

The network architecture is reported below:

|#  |Layers                        |Params |
|---|------------------------------|-------|
|1  | DilatedResidualBlock         |936    |
|2  | DownSample                   |0      |
|3  | DilatedResidualBlock         |5.7 K  |
|4  | DownSample                   |0      |
|5  | DilatedResidualBlock         |22.8 K |
|6  | DownSample                   |0      |
|7  | DilatedResidualBlock         |90.6 K |
|8  | DownSample                   |0      |
|9  | GLUBlock                     |27.1 K |
|10 | GLUBlock                     |27.1 K |
|11 | GLUBlock                     |27.1 K |
|12 | GLUBlock                     |27.1 K |
|13 | GLUBlock                     |27.1 K |
|14 | GLUBlock                     |27.1 K |
|15 | UpSample                     |0      |
|16 | DilatedResidualBlock         |86.5 K |
|17 | UpSample                     |0      |
|18 | DilatedResidualBlock         |36.1 K |
|19 | UpSample                     |0      |
|20 | DilatedResidualBlock         |9.1 K  |
|21 | UpSample                     |0      |
|22 | DilatedResidualBlock         |2.3 K  |
|23 | ConvTranspose1d              |9      |
|24 | Tanh                         |0      |

Downsampling layers halve the input size via max-pooling, whereas upsampling layers double it through linear 
interpolation. The dilation rate grows as a power of two $2^{j}$ with every $j$-th GLU block in the bottleneck, 
$j=0, ..., 5$.

The neural network is trained or 250,000 steps using RAdam and mini-batches of 128 randomly-sampled waveform segments. 

## Inference

First, download the pretrained [model weights](https://polimi365-my.sharepoint.com/:u:/g/personal/10391311_polimi_it/EeMrt4er9jJGg1ksJwoK-gEBBXG6ReI1RCXlwAGAy8m9iw?download=1) and save the checkpoint file in the `pretrained_models` folder.

Then, download the [exemplary test set](https://polimi365-my.sharepoint.com/:u:/g/personal/10391311_polimi_it/EXHNgqlP101LgUlhp9nB_wgBH20dcxsLRemGBG3TyJCAng?download=1) and extract it in the project's directory. 

If needed, update the `inference` fields in `config.yaml` according to your folder structure.

Finally, run 
```
$ python inference.py
```

## Exemplary test set
The **exemplary test set** set has the same folder structure as the yet-to-be-released **blind test set** (see [Challenge rules](https://github.com/polimi-ispl/2024-music-plc-challenge/blob/main/README.md)).

Participants may want to use the **exemplary test set** as a blueprint for preparing their custom test set. 

The **exemplary test set** contains 
- a folder containing lossy wav files, e.g, `lossy/Medley-solos-DB_test-0_0c83bc70-70ac-5e42-fc90-099e11e46287.wav`
- a directory containing packet loss traces in txt format, e.g., `traces/Medley-solos-DB_test-0_0c83bc70-70ac-5e42-fc90-099e11e46287.txt`
- a metadata file listing the test file names (without extension), see `meta.txt`

**Note:** To create a custom test set, participants are encouraged to use the packet traces from the INTERSPEECH 2022 Audio Deep 
Packet Loss Concealment Challenge, so as to simulate a loss schedule similar to that of the blind test set. 

🔗 Trace files are available [here](https://github.com/microsoft/PLC-Challenge).


## Training

To train PARCnet-IS² from scratch, run

```
$ python train.py
```

First, however, consider updating the following fields in `config.yaml` in order to specify the path to directory containing the training audio clips and the corresponding metadata file, respectively.

```
path:
  source_audio_dir:   "path/to/training/audio/directory"
  meta:               "path/to/training/metadata.csv"
```

----------

As is, the training code is meant to work with [Medley-solos-DB](https://doi.org/10.5281/zenodo.1344102).

Medley-solos-DB has the following folder structure:
```
medley-solos-db/
    - content.csv
    - Medley-solos-DB_test-0_04cb309e-9d0d-5486-f8f8-b9bd6714c359.wav
    - Medley-solos-DB_test-0_0e5669a9-9f3c-5fa3-fc80-44dc905d16f2.wav 
    - Medley-solos-DB_test-0_14b11cf0-680f-5399-faff-08149abfbcea.wav
    - ...
```

The metadata file `content.csv` has the following syntax:
```
subset,instrument,instrument_id,song_id,uuid4
test,clarinet,0,000,0e4371ac-1c6a-51ab-fdb7-f8abd5fbf1a3
test,clarinet,0,000,33383119-fd64-59c1-f596-d1a23e8a0eff
test,clarinet,0,000,b2b7a288-e169-5642-fced-b509c06b11fc
...
training,clarinet,0,139,163fd2b1-8e98-515a-f501-4742cc6d066f
...
validation,clarinet,0,200,a8bfed6b-3005-5059-f46d-4f4c5e672850
...
```

To use a different dataset other than Medley-solos-DB, make sure to modify **line 70** in `data.py`

Moreover, if your metadata does not have a column named `subset`, please modify **line 52** in `data.py` 

## Full list of changes
Here are the changes with respect to the original PARCnet model:

|                                     | Original PARCnet | PARCnet-IS² |
|-------------------------------------|------------------|-------------|
| Sampling rate                       | 32 kHz           | 44.1 kHz    |
| Packet size                         | 320 samples      | 512 samples |
| Extra prediction length             | 80 samples       | 256 samples |
| Cross-fade within burst loss        | ×                | ✓           |
| AR model order                      | 128              | 256         |
| AR valid context                    | 100 ms           | 92.8 ms     |
| AR safety range*                    | None             | [-1.5, 1.5] |
| Time-domain training loss           | L²-loss          | L¹-loss     |
| Neural contribution fade-in length  | 16 samples       | 64 samples  |
| Neural network valid context        | 7 packets        | 8 packets   |

\* The AR model yields zeros if the predicted signal takes values outside the safety range, if any.

📝 For additional details, please refer to the [original paper](https://doi.org/10.1109/OJSP.2023.3343318).

## License
- The code contained in this repository is released under CC BY 4.0 &nbsp; <img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by.svg" width="73">

- The available PARCnet weights are released under CC BY-NC-SA 4.0 &nbsp; <img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-nc-sa.svg" width="75">

## Citation
If you use PARCnet or PARCnet-IS² in your work, please consider citing
```
@ARTICLE{mezza2024hybrid,
  author={Mezza, Alessandro Ilic and Amerena, Matteo and Bernardini, Alberto and Sarti, Augusto},
  journal={IEEE Open Journal of Signal Processing}, 
  title={Hybrid Packet Loss Concealment for Real-Time Networked Music Applications}, 
  year={2024},
  volume={5},
  number={},
  pages={266--273},
  doi={10.1109/OJSP.2023.3343318}
}
```

## Contacts
For questions, open an issue or contact [music.plc.challenge.2024@gmail.com](mailto:music.plc.challenge.2024@gmail.com)
