import collections
import os
import zlib

import ctranslate2
import numpy as np
import tokenizers

from .feature_extractor import FeatureExtractor


class Segment(collections.namedtuple("Segment", ("start", "end", "text"))):
    pass


class AudioInfo(
    collections.namedtuple("AudioInfo", ("language", "language_probability"))
):
    pass


class TranscriptionOptions(
    collections.namedtuple(
        "TranscriptionOptions",
        (
            "beam_size",
            "best_of",
            "patience",
            "log_prob_threshold",
            "no_speech_threshold",
            "compression_ratio_threshold",
            "condition_on_previous_text",
            "temperatures",
        ),
    )
):
    pass


class WhisperModel:
    def __init__(
        self,
        model_path,
        device="auto",
        compute_type="default",
        cpu_threads=0,
    ):
        """Initializes the Whisper model.

        Args:
          model_path: Path to the converted model.
          device: Device to use for computation ("cpu", "cuda", "auto").
          compute_type: Type to use for computation.
            See https://opennmt.net/CTranslate2/quantization.html.
          cpu_threads: Number of threads to use when running on CPU (4 by default).
            A non zero value overrides the OMP_NUM_THREADS environment variable.
        """
        self.model = ctranslate2.models.Whisper(
            model_path,
            device=device,
            compute_type=compute_type,
            intra_threads=cpu_threads,
        )

        self.feature_extractor = FeatureExtractor()
        self.decoder = tokenizers.decoders.ByteLevel()

        with open(os.path.join(model_path, "vocabulary.txt"), encoding="utf-8") as vocab_file:
            self.ids_to_tokens = [line.rstrip("\n") for line in vocab_file]
            self.tokens_to_ids = {
                token: i for i, token in enumerate(self.ids_to_tokens)
            }

        self.eot_id = self.tokens_to_ids["<|endoftext|>"]
        self.timestamp_begin_id = self.tokens_to_ids["<|notimestamps|>"] + 1
        self.input_stride = 2
        self.time_precision = 0.02
        self.max_length = 448

    def transcribe(
        self,
        audio_bytes: bytes,
        language=None,
        beam_size=5,
        best_of=5,
        patience=1,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
    ):
        """Transcribes an input file.

        Arguments:
          input_file: Path to the input file or a file-like object.
          language: The language spoken in the audio. If not set, the language will be
            detected in the first 30 seconds of audio.
          beam_size: Beam size to use for decoding.
          best_of: Number of candidates when sampling with non-zero temperature.
          patience: Beam search patience factor.
          temperature: Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `logprob_threshold`.
          compression_ratio_threshold: If the gzip compression ratio is above this value,
            treat as failed.
          log_prob_threshold: If the average log probability over sampled tokens is
            below this value, treat as failed.
          no_speech_threshold: If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `logprob_threshold`,
            consider the segment as silent.
          condition_on_previous_text: If True, the previous output of the model is provided
            as a prompt for the next window; disabling may make the text inconsistent across
            windows, but the model becomes less prone to getting stuck in a failure loop,
            such as repetition looping or timestamps going out of sync.

        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of AudioInfo
        """
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32678.0
        features = self.feature_extractor(audio)

        if language is None:
            segment = self.get_segment(features)
            input = self.get_input(segment)
            results = self.model.detect_language(input)
            language_token, language_probability = results[0][0]
            language = language_token[2:-2]
        else:
            language_probability = 1

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
        )

        segments = self.generate_segments(features, language, options)

        audio_info = AudioInfo(
            language=language,
            language_probability=language_probability,
        )

        return segments, audio_info

    def generate_segments(self, features, language, options):
        tokenized_segments = self.generate_tokenized_segments(
            features, language, options
        )

        for start, end, tokens in tokenized_segments:
            text = self.decode_text_tokens(tokens)
            if not text.strip():
                continue

            yield Segment(
                start=start,
                end=end,
                text=text,
            )

    def generate_tokenized_segments(self, features, language, options):
        num_frames = features.shape[-1]
        offset = 0
        all_tokens = []
        prompt_reset_since = 0

        while offset < num_frames:
            time_offset = offset * self.feature_extractor.time_per_frame
            segment = self.get_segment(features, offset)
            segment_duration = segment.shape[-1] * self.feature_extractor.time_per_frame

            previous_tokens = all_tokens[prompt_reset_since:]
            prompt = self.get_prompt(language, previous_tokens)
            result, temperature = self.generate_with_fallback(segment, prompt, options)

            if (
                result.no_speech_prob > options.no_speech_threshold
                and result.scores[0] < options.log_prob_threshold
            ):
                offset += segment.shape[-1]
                continue

            tokens = result.sequences_ids[0]

            consecutive_timestamps = [
                i
                for i in range(len(tokens))
                if i > 0
                and tokens[i] >= self.timestamp_begin_id
                and tokens[i - 1] >= self.timestamp_begin_id
            ]

            if len(consecutive_timestamps) > 0:
                last_slice = 0
                for i, current_slice in enumerate(consecutive_timestamps):
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                        sliced_tokens[0] - self.timestamp_begin_id
                    )
                    end_timestamp_position = sliced_tokens[-1] - self.timestamp_begin_id
                    start_time = (
                        time_offset + start_timestamp_position * self.time_precision
                    )
                    end_time = (
                        time_offset + end_timestamp_position * self.time_precision
                    )

                    last_in_window = i + 1 == len(consecutive_timestamps)

                    # Include the last timestamp so that all tokens are included in a segment.
                    if last_in_window:
                        sliced_tokens.append(tokens[current_slice])

                    yield start_time, end_time, sliced_tokens
                    last_slice = current_slice

                last_timestamp_position = (
                    tokens[last_slice - 1] - self.timestamp_begin_id
                )
                offset += last_timestamp_position * self.input_stride
                all_tokens.extend(tokens[: last_slice + 1])

            else:
                duration = segment_duration
                timestamps = [
                    token for token in tokens if token >= self.timestamp_begin_id
                ]
                if len(timestamps) > 0 and timestamps[-1] != self.timestamp_begin_id:
                    last_timestamp_position = timestamps[-1] - self.timestamp_begin_id
                    duration = last_timestamp_position * self.time_precision

                yield time_offset, time_offset + duration, tokens

                offset += segment.shape[-1]
                all_tokens.extend(tokens)

            if not options.condition_on_previous_text or temperature > 0.5:
                prompt_reset_since = len(all_tokens)

    def decode_text_tokens(self, tokens):
        text_tokens = [
            self.ids_to_tokens[token] for token in tokens if token < self.eot_id
        ]

        return self.decoder.decode(text_tokens)

    def generate_with_fallback(self, segment, prompt, options):
        features = self.get_input(segment)
        result = None
        final_temperature = None

        for temperature in options.temperatures:
            if temperature > 0:
                kwargs = {
                    "beam_size": 1,
                    "num_hypotheses": options.best_of,
                    "sampling_topk": 0,
                    "sampling_temperature": temperature,
                }
            else:
                kwargs = {
                    "beam_size": options.beam_size,
                    "patience": options.patience,
                }

            final_temperature = temperature
            result = self.model.generate(
                features,
                [prompt],
                max_length=self.max_length,
                return_scores=True,
                return_no_speech_prob=True,
                **kwargs,
            )[0]

            tokens = result.sequences_ids[0]
            text = self.decode_text_tokens(tokens)
            compression_ratio = get_compression_ratio(text)

            if (
                compression_ratio <= options.compression_ratio_threshold
                and result.scores[0] >= options.log_prob_threshold
            ):
                break

        return result, final_temperature

    def get_prompt(self, language, previous_tokens):
        prompt = []

        if previous_tokens:
            prompt.append(self.tokens_to_ids["<|startofprev|>"])
            prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

        prompt += [
            self.tokens_to_ids["<|startoftranscript|>"],
            self.tokens_to_ids["<|%s|>" % language],
            self.tokens_to_ids["<|transcribe|>"],
        ]

        return prompt

    def get_segment(self, features, offset=0):
        if offset > 0:
            features = features[:, offset:]

        num_frames = features.shape[-1]
        required_num_frames = self.feature_extractor.nb_max_frames

        if num_frames > required_num_frames:
            features = features[:, :required_num_frames]
        elif num_frames < required_num_frames:
            pad_widths = [(0, 0), (0, required_num_frames - num_frames)]
            features = np.pad(features, pad_widths)

        features = np.ascontiguousarray(features)
        return features

    def get_input(self, segment):
        segment = np.expand_dims(segment, 0)
        segment = ctranslate2.StorageView.from_array(segment)
        return segment


def get_compression_ratio(text):
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))
