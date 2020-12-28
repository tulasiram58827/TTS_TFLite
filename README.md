This repository provides a collection of widely popular text-to-speech (TTS) models in TensorFlow Lite (TFLite). These models primarily come from two repositories - [TTS](https://github.com/mozilla/TTS) and [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS). We provide end-to-end Colab Notebooks that show the model conversion and inference process using TFLite. This includes converting PyTorch models to TFLite as well. 

TTS is a two-step process - first you generate a MEL spectrogram using a TTS model and then you pass it to a VOCODER for generating the audio waveform.  We include both of these models inside this repository.  

**Note** that these models are trained on [LJSpeech dataset](https://www.tensorflow.org/datasets/catalog/ljspeech).

[Here’s a sample result](https://storage.googleapis.com/demo-experiments/demo_tts.wav) (with Fastspeech2 and MelGAN) for the text “Bill got in the habit of asking himself".

## Models Included


- TTS:
    - [x] [Tacotron2](https://github.com/NVIDIA/tacotron2)
    - [x] [Fastspeech2](https://arxiv.org/abs/2006.04558)
    - [ ] [Glow TTS](https://arxiv.org/abs/2005.11129)*
- VOCODER:
    - [x] [MelGAN](https://arxiv.org/abs/1910.06711)
    - [x] [Multi-Band MelGAN](https://arxiv.org/abs/2005.05106) (MB MelGAN)
    - [x] [Parallel WaveGAN](https://arxiv.org/abs/1910.11480)

In the future, we may add more models.

<small> *Currently, conversion of the Glow TTS model is unavailable (refer to the issue [here](https://github.com/mozilla/TTS/issues/608)). </small>

## About the Notebooks
- `End_to_End_TTS.ipynb`: This notebook allows you to load up different TTS and VOCODER models (enlisted above) and to perform inference. 
- `MelGAN_TFLite.ipynb`: Shows the model conversion process of MelGAN. 
- `Parallel_WaveGAN_TFLite.ipynb`: Shows the model conversion process of Parallel WaveGAN. 

Model conversion processes for Tacotron2, Fastspeech2, and Multi-Band MelGAN are available via the following notebooks:

- [Tacotron2 & Multi-Band MelGAN](https://colab.research.google.com/github/mozilla/TTS/blob/master/notebooks/DDC_TTS_and_MultiBand_MelGAN_TFLite_Example.ipynb)
- [Fastspeech2](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/notebooks/TensorFlowTTS_FastSpeech_with_TFLite.ipynb)
## Model Benchmarks

After converting to TFLite, we used the [Benchmark tool](https://www.tensorflow.org/lite/performance/measurement) in order to report performance metrics of the various models such as inference latency, peak memory usage. We used Redmi K20 for this purpose. For all the experiments we kept the number of threads to one and we used the CPU of Redmi K20 and no other hardware accelerator. 

| **Model**        | **Quantization** | **Model Size (MB)** | **Average Inference Latency (seconds)** | **Memory Footprint (MB)** |
| ---------------- | ---------------- | ------------------- | --------------------------------------- | ------------------------- |
| Parallel WaveGAN | Dynamic-range    | 5.7                 | 0.04                                    | 31.5                      |
| Parallel WaveGAN | Float16          | 3.2                 | 0.05                                    | 34                        |
| MelGAN           | Dynamic-range    | 17                  | 0.51                                    | 81                        |
| MelGAN           | Float16          | 8.3                 | 0.52                                    | 89                        |
| MB MelGAN        | Dynamic-range    | 17                  | 0.02                                    | 17                        |
| Tacotron2        | Dynamic-range    | 30.1                | 1.66                                    | 75                        |
| FastSpeech2      | Dynamic-range    | 30                  | 0.11s                                   | 55                        |

**Notes**:

- All the models above support dynamic shaped inputs. However, benchmarking dynamic input size MelGAN models is not currently supported. So to benchmark those models we used inputs of shape (100, 80).
- Benchmarking of the Fastspeech2 models is currently erroring out. 


## References
- [Dynamic-range quantization in TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_quant)
- [Float16 quantization in TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_float16_quant)
