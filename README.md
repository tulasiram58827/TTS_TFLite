This repository provides a collection of widely popular text-to-speech (TTS) models in TensorFlow Lite (TFLite). These models primarily come from two repositories - [TTS](https://github.com/mozilla/TTS) and [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS). We provide end-to-end Colab Notebooks that show the model conversion and inference process using TFLite. This includes converting PyTorch models to TFLite as well. 

TTS is a two-step process - first you generate a MEL spectrogram using a TTS model and then you pass it to a VOCODER for generating the audio waveform.  We include both of these models inside this repository.  

**Note** that these models are trained on [LJSpeech dataset](https://www.tensorflow.org/datasets/catalog/ljspeech).

[Here’s a sample result](https://storage.googleapis.com/demo-experiments/demo_tts.wav) (with Fastspeech2 and MelGAN) for the text “Bill got in the habit of asking himself".

## Models Included


- TTS:
    - [x] [Tacotron2](https://github.com/NVIDIA/tacotron2)
    - [x] [Fastspeech2](https://arxiv.org/abs/2006.04558)
    - [x] [Forward Tacotron](https://github.com/as-ideas/ForwardTacotron)
    - [ ] [Glow TTS](https://arxiv.org/abs/2005.11129)*
    - [ ] [Transformer TTS](https://arxiv.org/abs/1809.08895)
- VOCODER:
    - [x] [MelGAN](https://arxiv.org/abs/1910.06711)
    - [x] [Multi-Band MelGAN](https://arxiv.org/abs/2005.05106) (MB MelGAN)
    - [x] [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)
    - [x] [HiFi-GAN](https://arxiv.org/pdf/2010.05646.pdf)

In the future, we may add more models.

<small> *Currently, conversion of the Glow TTS model is unavailable (refer to the issue [here](https://github.com/pytorch/pytorch/issues/50009)). </small>

Currently, **Forward Tacotron** only supports ONNX Conversion. There is a problem while converting to TensorFlow Graph Format. (Refer to this [issue](https://github.com/onnx/onnx-tensorflow/issues/853) for more details).

**Notes:**

- Training data used for HiFi-GAN (MEL spectogram generation) is different w.r.t other models like Tacotron2, FastSpech2. So it is not compatible with the other architectures available inside this repo.
- If you want to use HiFi-GAN in end-to-end scenario you can refer to this [notebook](https://github.com/jaywalnut310/glow-tts/blob/master/inference_hifigan.ipynb). In future we are planning to make it compatible with other architectures and add it in our [end-to-end notebook](https://github.com/tulasiram58827/TTS_TFLite/blob/main/End_to_End_TTS.ipynb). Stay tuned!

## About the Notebooks
- `End_to_End_TTS.ipynb`: This notebook allows you to load up different TTS and VOCODER models (enlisted above) and to perform inference. 
- `MelGAN_TFLite.ipynb`: Shows the model conversion process of MelGAN. 
- `Parallel_WaveGAN_TFLite.ipynb`: Shows the model conversion process of Parallel WaveGAN. 
- `HiFi-GAN.ipynb`: Shows the model conversion process of HiFi-GAN.
- `Forward_Tacotron_PyTorch_TFLite.ipynb` : Converts Forward Tacotron model to ONNX. In future it will be updated to support TFLite conversion.

Model conversion processes for Tacotron2, Fastspeech2, and Multi-Band MelGAN are available via the following notebooks:

- [Tacotron2 & Multi-Band MelGAN](https://colab.research.google.com/github/mozilla/TTS/blob/master/notebooks/DDC_TTS_and_MultiBand_MelGAN_TFLite_Example.ipynb)
- [Fastspeech2](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/notebooks/TensorFlowTTS_FastSpeech_with_TFLite.ipynb)
## Model Benchmarks

After converting to TFLite, we used the [Benchmark tool](https://www.tensorflow.org/lite/performance/measurement) in order to report performance metrics of the various models such as inference latency, peak memory usage. We used Redmi K20 for this purpose. For all the experiments we kept the number of threads to one and we used the CPU of Redmi K20 and no other hardware accelerator. 

| **Model**        | **Quantization** | **Model Size (MB)** | **Average Inference Latency (sec)** | **Memory Footprint (MB)** |
| ---------------- | ---------------- | :-----------------: | :----------------------------------:| :-----------------------: |
| Parallel WaveGAN | Dynamic-range    | 5.7                 | 0.04                                | 31.5                      |
| Parallel WaveGAN | Float16          | 3.2                 | 0.05                                | 34                        |
| MelGAN           | Dynamic-range    | 17                  | 0.51                                | 81                        |
| MelGAN           | Float16          | 8.3                 | 0.52                                | 89                        |
| MB MelGAN        | Dynamic-range    | 17                  | 0.02                                | 17                        |
| HiFi-GAN         | Dynamic-range    | 3.5                 | 0.0015                              | 9.88                      |
| HiFi-GAN         | Float16          | 2.9                 | 0.0036                              | 20.3                      | 
| Tacotron2        | Dynamic-range    | 30.1                | 1.66                                | 75                        |
| Fastspeech2      | Dynamic-range    | 30                  | 0.11                                | 55                        |

**Notes**:

- All the models above support dynamic shaped inputs. However, benchmarking dynamic input size MelGAN models is not currently supported. So to benchmark those models we used inputs of shape (100, 80).
- Similary for Fastspeech2 benchmarking dynamic input size model is erroring out. So to benchmark we used inputs of shape (1, 50) where 50 represents number of tokens. [This issue thread](https://github.com/tensorflow/tensorflow/issues/45986) provides more details. 

## 🔈 Audio Samples

All combination of samples are available in `audio_samples` folder. To listen directly without downloading refer to this [Sound Cloud](https://soundcloud.com/tulasi-ram-887761209) folder.

## References
- [Dynamic-range quantization in TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_quant)
- [Float16 quantization in TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_float16_quant)
