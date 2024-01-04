# Faster Whisper transcription with CTranslate2

This repository demonstrates how to implement the Whisper transcription using [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which is a fast inference engine for Transformer models.

This implementation is about 4 times faster than [openai/whisper](https://github.com/openai/whisper) for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.

## Installation

```bash
pip install -e .[conversion]
```

The model conversion requires the modules `transformers` and `torch` which are installed by the `[conversion]` requirement. Once a model is converted, these modules are no longer needed and the installation could be simplified to:

```bash
pip install -e .
```

## Usage

### Model conversion

A Whisper model should be first converted into the CTranslate2 format. For example the command below converts the "medium" Whisper model and saves the weights in FP16:

```bash
ct2-transformers-converter --model openai/whisper-medium --output_dir whisper-medium-ct2 --quantization float16
```

If needed, models can also be converted from the code. See the [conversion API](https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html).

### Transcription

```python
from faster_whisper import WhisperModel

model_path = "whisper-medium-ct2/"

# Run on GPU with FP16
model = WhisperModel(model_path, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_path, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_path, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%ds -> %ds] %s" % (segment.start, segment.end, segment.text))
```

## Comparing performance against openai/whisper

If you are comparing the performance against [openai/whisper](https://github.com/openai/whisper), you should make sure to use the same settings in both frameworks. In particular:

* In openai/whisper, `model.transcribe` uses a beam size of 1 by default. A different beam size will have an important impact on performance so make sure to use the same.
* When running on CPU, make sure to set the same number of threads. Both frameworks will read the environment variable `OMP_NUM_THREADS`, which can be set when running your script:

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```
