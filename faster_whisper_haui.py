import shutil
import tarfile
import array
import io
import wave
import time
import logging

from typing import TYPE_CHECKING, Final, Literal
from collections.abc import Generator, Iterable
from functools import partial
from dataclasses import dataclass, field
from urllib.request import urlopen
from pathlib import Path
from tkinter import ttk
import tkinter
from tkinter import filedialog

if TYPE_CHECKING:
    from webrtc_noise_gain import AudioProcessor

from faster_whisper import WhisperModel

VERSION: Final = "2024.01.04"

TK_READONLY: Final = "readonly"

logging.basicConfig(
    filename="debug.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# NOTE: Adapted to match Home Assistant configuration/logic from:
#       https://github.com/rhasspy/wyoming-faster-whisper/ 
#       https://github.com/home-assistant/core/blob/dev/homeassistant/components/assist_pipeline/

class AudioBuffer:
    """Fixed-sized audio buffer with variable internal length."""

    def __init__(self, maxlen: int) -> None:
        """Initialize buffer."""
        self._buffer = bytearray(maxlen)
        self._length = 0

    @property
    def length(self) -> int:
        """Get number of bytes currently in the buffer."""
        return self._length

    def clear(self) -> None:
        """Clear the buffer."""
        self._length = 0

    def append(self, data: bytes) -> None:
        """Append bytes to the buffer, increasing the internal length."""
        data_len = len(data)
        if (self._length + data_len) > len(self._buffer):
            raise ValueError("Length cannot be greater than buffer size")

        self._buffer[self._length : self._length + data_len] = data
        self._length += data_len

    def bytes(self) -> bytes:
        """Convert written portion of buffer to bytes."""
        return bytes(self._buffer[: self._length])

    def __len__(self) -> int:
        """Get the number of bytes currently in the buffer."""
        return self._length

    def __bool__(self) -> bool:
        """Return True if there are bytes in the buffer."""
        return self._length > 0


@dataclass(frozen=True, slots=True)
class ProcessedAudioChunk:
    """Processed audio chunk and metadata."""

    audio: bytes
    """Raw PCM audio @ 16Khz with 16-bit mono samples"""

    timestamp_ms: int
    """Timestamp relative to start of audio stream (milliseconds)"""

    is_speech: bool | None
    """True if audio chunk likely contains speech, False if not, None if unknown"""


class App(tkinter.Tk):
    # UI defaults
    DEFAULT_THEME: Final = "vista"
    GEO_W: Final = 800
    GEO_H: Final = 600
    CONFIG_GEO_W: Final = int(GEO_W * 0.75)
    CONFIG_GEO_H: Final = int(GEO_H * 0.75)
    CONFIG_GEO_OFFSET_X: Final = 0
    CONFIG_GEO_OFFSET_Y: Final = 30
    LABEL_WIDTH: Final = 33
    ENTRY_WIDTH: Final = 66
    ROW_WIDTH: Final = (LABEL_WIDTH + ENTRY_WIDTH)
    ROW_PADDING_X: Final = 6
    ROW_PADDING_Y: Final = 6
    ROW_PADDING: Final = { "padx": ROW_PADDING_X, "pady": ROW_PADDING_Y }

    # ms
    ROUND_SECONDS: Final = 3
    ROUND_PROBABILITY: Final = 4

    DATA_DIR: Final = Path(".", "data")
    URL_FORMAT: Final = "https://github.com/rhasspy/models/releases/download/v1.0/asr_faster-whisper-{model}.tar.gz"
    MODEL_BIN_FILENAME: Final = "model.bin"

    # Defaults same as Home Assistant
    # https://github.com/home-assistant/addons/blob/133709534a0bc853ee93a2315657b59caa67ca0e/whisper/config.yaml#L15
    # https://github.com/home-assistant/addons/blob/133709534a0bc853ee93a2315657b59caa67ca0e/whisper/rootfs/etc/s6-overlay/s6-rc.d/whisper/run
    # https://github.com/rhasspy/wyoming-faster-whisper/blob/a21720ce3f9a0aefd4a52286e15735825a4805be/wyoming_faster_whisper/__main__.py#L20
    # https://esphome.io/components/voice_assistant
    MODELS: Final = ["tiny-int8", "tiny", "base", "base-int8", "small-int8", "small", "medium-int8"]
    LANGUAGES: Final = ["auto", "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh"]
    BEAM_SIZES: Final = list(range(1, 11))
    NOISE_SUPPR_LEVELS: Final = list(range(0, 5))
    AUTO_GAINS: Final = list(range(0, 32))

    DEFAULT_MODEL: Final = "tiny-int8"
    DEFAULT_LANGUAGE: Final = "en"
    DEFAULT_BEAM_SIZE: Final = 1
    DEFAULT_DEVICE: Final = "cpu"
    DEFAULT_COMPUTE_TYPE: Final = "default"
    # integer - The noise suppression level to apply to the assist pipeline. Between 0 and 4 inclusive. Defaults to 0 (disabled).
    DEFAULT_NOISE_SUPPRESSION_LEVEL: Final = 0
    # dBFS - Auto gain level to apply to the assist pipeline. Between 0dBFS and 31dBFS inclusive. Defaults to 0 (disabled).
    DEFAULT_AUTO_GAIN: Final = 0
    # float - Volume multiplier to apply to the assist pipeline. Must be larger than 0. Defaults to 1 (disabled).
    DEFAULT_VOLUME_MULTIPLIER: Final = 1.0

    AUDIO_PROCESSOR_SAMPLES: Final = 160  # 10 ms @ 16 Khz
    AUDIO_PROCESSOR_BYTES: Final = AUDIO_PROCESSOR_SAMPLES * 2  # 16-bit samples
    SAMPLE_RATE: Final = 16000
    SAMPLE_WIDTH: Final = 2
    SAMPLE_CHANNELS: Final = 1
    SAMPLES_PER_CHUNK: Final = 1024

    chunking_enabled: bool = True
    audio_processor_buffer: AudioBuffer = field(init=False, repr=False)
    audio_processor = None

    # Set to true whenever a config change requires reloading
    # Triggers a full restart on config window exit
    needs_reloading: bool = False

    def __init__(self) -> None:
        super().__init__()

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.geometry(f"{self.GEO_W}x{self.GEO_H}")
        self.title(f"faster-whisper-HAUI - v{VERSION}")
        self.configure(bg="white")

        style = ttk.Style()

        style.theme_use(self.DEFAULT_THEME if self.DEFAULT_THEME in style.theme_names() else "default")
        style.configure("TButton", padding=4, font=("Arial", 11, "normal"))
        style.configure("TLabel", padding=3, font=("Arial", 11, "normal"))
        style.configure("TEntry", padding=3, font=("Arial", 11, "normal"))
        style.configure("TCombobox", padding=3, font=("Arial", 11, "normal"))
        style.configure("TSpinbox", padding=3, font=("Arial", 11, "normal"))

        self.last_used_dir = tkinter.StringVar(value=str(Path(".")))
        self.audio_filepath = tkinter.StringVar(value=str(Path(".", "audio.wav")))

        available_models: list[str] = []

        for model in self.MODELS:
            if self.get_model_dir(model) is not None:
                available_models.append(model)

        logging.debug(f"Models found: {available_models}.")

        top_container_frame: ttk.Frame = ttk.Frame(self)
        model_cb_state: bool = tkinter.DISABLED
        model_cb_default: str = "none available" 

        if len(available_models) >= 1:
            model_cb_state = TK_READONLY
            model_cb_default = available_models[0]
        else:
            available_models = [model_cb_default]

        model_row: ttk.Frame = ttk.Frame(top_container_frame)
        self.model: ttk.Combobox = ttk.Combobox(model_row, values=available_models, state=model_cb_state)
        model_label: ttk.Label = ttk.Label(model_row, text="Model:", width=self.LABEL_WIDTH, anchor=tkinter.W)

        language_row: ttk.Frame = ttk.Frame(top_container_frame)
        self.language: ttk.Combobox = ttk.Combobox(language_row, values=self.LANGUAGES, state=TK_READONLY)
        language_label: ttk.Label = ttk.Label(language_row, text="Language:", width=self.LABEL_WIDTH, anchor=tkinter.W)

        beam_size_row: ttk.Frame = ttk.Frame(top_container_frame)
        self.beam_size: ttk.Combobox = ttk.Combobox(beam_size_row, values=self.BEAM_SIZES, state=TK_READONLY)
        beam_size_label: ttk.Label = ttk.Label(beam_size_row, text="Beam Size:", width=self.LABEL_WIDTH, anchor=tkinter.W)

        noise_suppression_level_row: ttk.Frame = ttk.Frame(top_container_frame)
        self.noise_suppression_level: ttk.Combobox = ttk.Combobox(noise_suppression_level_row, values=self.NOISE_SUPPR_LEVELS, state=TK_READONLY)
        noise_suppression_level_label: ttk.Label = ttk.Label(noise_suppression_level_row, text="Noise Suppression Level:", width=self.LABEL_WIDTH, anchor=tkinter.W)

        auto_gain_row: ttk.Frame = ttk.Frame(top_container_frame)
        self.auto_gain: ttk.Combobox = ttk.Combobox(auto_gain_row, values=self.AUTO_GAINS, state=TK_READONLY)
        auto_gain_label: ttk.Label = ttk.Label(auto_gain_row, text="Auto Gain (dBFS):", width=self.LABEL_WIDTH, anchor=tkinter.W)

        volume_multiplier_row: ttk.Frame = ttk.Frame(top_container_frame)
        self.volume_multiplier = ttk.Spinbox(volume_multiplier_row, from_=0.0, to=5.0, increment=0.1, wrap=True)
        volume_multiplier_label: ttk.Label = ttk.Label(volume_multiplier_row, text="Volume Multiplier:", width=self.LABEL_WIDTH, anchor=tkinter.W)

        separator: ttk.Separator = ttk.Separator(top_container_frame, orient=tkinter.HORIZONTAL)

        audio_fp_row: ttk.Frame = ttk.Frame(top_container_frame)
        audio_fp_label: ttk.Label = ttk.Label(audio_fp_row, textvariable=self.audio_filepath, background="#eaeaea")
        audio_fp_button: ttk.Button = ttk.Button(audio_fp_row, text="Browse Audio File", command=self.browse_audio_filepath, width=self.LABEL_WIDTH)

        self.model.set(model_cb_default)
        self.language.set(self.DEFAULT_LANGUAGE)
        self.beam_size.set(self.DEFAULT_BEAM_SIZE)
        self.noise_suppression_level.set(self.DEFAULT_NOISE_SUPPRESSION_LEVEL)
        self.auto_gain.set(self.DEFAULT_AUTO_GAIN)
        self.volume_multiplier.set(self.DEFAULT_VOLUME_MULTIPLIER)

        model_label.pack(side=tkinter.LEFT, padx=(0, self.ROW_PADDING_X))
        self.model.pack(side=tkinter.RIGHT, expand=tkinter.YES, fill=tkinter.X)
        language_label.pack(side=tkinter.LEFT, padx=(0, self.ROW_PADDING_X))
        self.language.pack(side=tkinter.RIGHT, expand=tkinter.YES, fill=tkinter.X)
        beam_size_label.pack(side=tkinter.LEFT, padx=(0, self.ROW_PADDING_X))
        self.beam_size.pack(side=tkinter.RIGHT, expand=tkinter.YES, fill=tkinter.X)
        noise_suppression_level_label.pack(side=tkinter.LEFT, padx=(0, self.ROW_PADDING_X))
        self.noise_suppression_level.pack(side=tkinter.RIGHT, expand=tkinter.YES, fill=tkinter.X)
        auto_gain_label.pack(side=tkinter.LEFT, padx=(0, self.ROW_PADDING_X))
        self.auto_gain.pack(side=tkinter.RIGHT, expand=tkinter.YES, fill=tkinter.X)
        volume_multiplier_label.pack(side=tkinter.LEFT, padx=(0, self.ROW_PADDING_X))
        self.volume_multiplier.pack(side=tkinter.RIGHT, expand=tkinter.YES, fill=tkinter.X)
        audio_fp_button.pack(side=tkinter.LEFT, padx=(0, self.ROW_PADDING_X))
        audio_fp_label.pack(side=tkinter.RIGHT, expand=tkinter.YES, fill=tkinter.X)

        model_row.pack(side=tkinter.TOP, fill=tkinter.X, **self.ROW_PADDING)
        language_row.pack(side=tkinter.TOP, fill=tkinter.X, **self.ROW_PADDING)
        beam_size_row.pack(side=tkinter.TOP, fill=tkinter.X, **self.ROW_PADDING)
        noise_suppression_level_row.pack(side=tkinter.TOP, fill=tkinter.X, **self.ROW_PADDING)
        auto_gain_row.pack(side=tkinter.TOP, fill=tkinter.X, **self.ROW_PADDING)
        volume_multiplier_row.pack(side=tkinter.TOP, fill=tkinter.X, **self.ROW_PADDING)
        separator.pack(side=tkinter.TOP, fill=tkinter.X, **self.ROW_PADDING)
        audio_fp_row.pack(side=tkinter.TOP, fill=tkinter.X, **self.ROW_PADDING)
        top_container_frame.pack(side=tkinter.TOP, expand=tkinter.YES, fill=tkinter.BOTH)

        mid_container_frame: ttk.Frame = ttk.Frame(self)
        self.transcribed_label: ttk.Label = ttk.Label(mid_container_frame, width=self.ROW_WIDTH, text='', wraplength=self.GEO_W - (self.ROW_PADDING_X * 2), justify=tkinter.LEFT)

        self.transcribed_label.pack(side=tkinter.TOP, fill=tkinter.BOTH, **self.ROW_PADDING)
        mid_container_frame.pack(side=tkinter.TOP, expand=tkinter.YES, fill=tkinter.BOTH)

        bot_container_frame: ttk.Frame = ttk.Frame(self)
        buttons_row: ttk.Frame = ttk.Frame(bot_container_frame)
        save_processed_button: ttk.Button = ttk.Button(buttons_row, text="Save Processed Audio", command=self.save_processed_audio)
        run_whisper: ttk.Button = ttk.Button(buttons_row, text="Run Whisper", command=self.run_whisper)
        edit_config_button: ttk.Button = ttk.Button(buttons_row, text="Edit Config", command=self.show_config_window)
        self.info_label: ttk.Label = ttk.Label(bot_container_frame, width=self.ROW_WIDTH, text="", background="#eaeaea")

        self.info_label.pack(side=tkinter.BOTTOM, fill=tkinter.X, **self.ROW_PADDING)
        buttons_row.pack(side=tkinter.BOTTOM, fill=tkinter.X, **self.ROW_PADDING)
        edit_config_button.pack(side=tkinter.LEFT, padx=(0, self.ROW_PADDING_X))
        run_whisper.pack(side=tkinter.RIGHT, padx=(0, self.ROW_PADDING_X))
        save_processed_button.pack(side=tkinter.RIGHT, padx=(0, self.ROW_PADDING_X))
        bot_container_frame.pack(side=tkinter.BOTTOM, expand=tkinter.YES, fill=tkinter.BOTH)

    def on_config_window_exit(self, config_window: tkinter.Toplevel) -> None:
        if self.needs_reloading:
            restart_app(self)
        else:
            config_window.destroy()

    def show_config_window(self) -> None:
        config_window = tkinter.Toplevel(self, padx=4, pady=4)
        x: int = self.winfo_x() + self.CONFIG_GEO_OFFSET_X
        y: int = self.winfo_y() + self.CONFIG_GEO_OFFSET_Y
        config_window.geometry(f"{self.CONFIG_GEO_W}x{self.CONFIG_GEO_H}+{x}+{y}")
        config_window.title("Config")
        config_window.protocol("WM_DELETE_WINDOW", partial(self.on_config_window_exit, config_window))

        # prevent interacting with root
        config_window.grab_set()
        config_window.focus_force()

        models_container_frame: ttk.Frame = ttk.Frame(config_window)

        for model in self.MODELS:
            model_row: ttk.Frame = ttk.Frame(models_container_frame)
            model_label: ttk.Label = ttk.Label(model_row, text=model, anchor=tkinter.W)
            model_button: ttk.Button = ttk.Button(model_row, text=f"Download {model}", command=partial(self.download_model, model))

            model_label.pack(side=tkinter.LEFT, expand=tkinter.YES, fill=tkinter.X)
            model_button.pack(side=tkinter.RIGHT, padx=(0, self.ROW_PADDING_X))

            if self.is_model_dir_valid(Path(self.DATA_DIR, model)):
                model_label_last_edit: ttk.Label = ttk.Label(model_row, text="(Found)", anchor=tkinter.W)

                model_label_last_edit.pack(side=tkinter.RIGHT)

            model_row.pack(side=tkinter.TOP, fill=tkinter.X, **self.ROW_PADDING)

        models_container_frame.pack(side=tkinter.TOP, expand=tkinter.YES, fill=tkinter.BOTH)

        bot_container_frame: ttk.Frame = ttk.Frame(config_window)
        buttons_row: ttk.Frame = ttk.Frame(bot_container_frame)
        reload_button: ttk.Button = ttk.Button(buttons_row, text="Reload", command=partial(restart_app, self))

        reload_button.pack(side=tkinter.RIGHT, padx=(0, self.ROW_PADDING_X))
        buttons_row.pack(side=tkinter.BOTTOM, fill=tkinter.X, **self.ROW_PADDING)
        bot_container_frame.pack(side=tkinter.BOTTOM, expand=tkinter.YES, fill=tkinter.BOTH)

    def set_info_label(self, msg: str, color: Literal["red", "orange", "yellow", "blue", "black"] = "black") -> None:
        if self.info_label:
            self.info_label.config(text=msg, foreground=color)

        if color == "red":
            logging.exception(msg)
        elif color == "orange":
            logging.error(msg)
        elif color == "yellow":
            logging.warn(msg)
        elif color == "blue":
            logging.info(msg)
        else:
            logging.debug(msg)

    def set_transcribed_label(self, msg: str) -> None:
        if self.transcribed_label:
            self.transcribed_label.config(text=msg, foreground="black")

        logging.info(f"STT OUTPUT: {msg}")

    def browse_audio_filepath(self) -> None:
        self.set_info_label("Loading audio file...")

        filename = filedialog.askopenfilename(parent=self, initialdir=self.last_used_dir.get(), title="Select Audio File", filetypes=[("WAV Files", "*.wav")])

        self.last_used_dir.set(Path(filename).parent)

        if filename and Path(filename).is_file():
            self.audio_filepath.set(filename)
            self.set_info_label(f"Loaded audio file {filename}.")
        else:
            self.audio_filepath.set("")
            self.set_info_label(f"No audio file loaded.", color="orange")

    def save_processed_audio(self) -> None:
        self.set_info_label(f"Running with noise_suppression_level: '{self.noise_suppression_level.get()}', auto_gain: '{self.auto_gain.get()}', volume_multiplier: '{self.volume_multiplier.get()}'", color="blue")
        self.update()

        try:
            start_time: int = time.time()
            filename: str = f"{self.audio_filepath.get().replace('.wav', '_p')}{time.monotonic_ns()}.wav"

            with wave.open(filename, "wb") as wav_writer:
                wav_writer.setframerate(16000)
                wav_writer.setsampwidth(2)
                wav_writer.setnchannels(1)

                for chunk in self.process_audio(self.process_audio_wav()):
                    wav_writer.writeframes(chunk.audio)

            self.set_info_label(f"Process: {round(time.time() - start_time, self.ROUND_SECONDS)} seconds.", color="blue")
        except:
            self.set_info_label(f"Failed to Process.", color="red")
            raise

    def run_whisper(self) -> None:
        self.set_info_label(f"Running with model: '{self.model.get()}', language: '{self.language.get()}', beam_size: '{self.beam_size.get()}', noise_suppression_level: '{self.noise_suppression_level.get()}', auto_gain: '{self.auto_gain.get()}', volume_multiplier: '{self.volume_multiplier.get()}'", color="blue")
        self.update()

        model: Path | None = self.get_model_dir(self.model.get())

        if model is None:
            self.set_info_label(f"Model '{self.model.get()}' not found/valid in '{model}'. Go to config to re-download it.", color="red")
            raise FileNotFoundError(model)

        whisper_model = WhisperModel(str(model), device=self.DEFAULT_DEVICE, compute_type=self.DEFAULT_COMPUTE_TYPE)

        try:
            start_time: int = time.time()
            audio_bytes: bytes = bytes()

            for chunk in self.process_audio(self.process_audio_wav()):
                audio_bytes += chunk.audio
            
            post_process_time: int = time.time()
            lang: str = self.language.get()
            segments, _info = whisper_model.transcribe(audio_bytes, beam_size=int(self.beam_size.get()), language=lang if lang != "auto" else None)
            text: str = " ".join(segment.text for segment in segments)

            self.set_transcribed_label(text)

            detected_lang: str = ""

            if lang == "auto":
                detected_lang = f" Detected '{_info.language}', with probability '{round(_info.language_probability, self.ROUND_PROBABILITY)}'."

            self.set_info_label(f"Process: {round(post_process_time - start_time, self.ROUND_SECONDS)} seconds | Transcribe: {round(time.time() - start_time, self.ROUND_SECONDS)} seconds.{detected_lang}", color="blue")
        except:
            self.set_info_label(f"Failed to Process+Transcribe.", color="red")
            raise

    def download_model(self, model) -> None:
        self.needs_reloading = True
        model_url = self.URL_FORMAT.format(model=model)
        model_dir = Path(self.DATA_DIR, model)

        self.set_info_label(f"Downloading '{model}' to '{self.DATA_DIR}' from '{model_url}'...", color="blue")
        self.update()

        if model_dir.is_dir():
            # Remove model directory if it already exists
            shutil.rmtree(model_dir)

        try:
            with urlopen(model_url) as response:
                with tarfile.open(mode="r|*", fileobj=response) as tar_gz:
                    self.set_info_label("Extracting model...", color="blue")
                    tar_gz.extractall(self.DATA_DIR)
        except:
            self.set_info_label(f"Failed to download and extract model", color="red")
            raise

    def get_model_dir(self, model: str) -> Path | None:
        model_dir: Path = Path(self.DATA_DIR, model)

        if self.is_model_dir_valid(model_dir):
            return model_dir
        else:
            return None

    def is_model_dir_valid(self, model_dir: str | Path) -> bool:
        if not model_dir:
            return False

        model_bin: Path = Path(model_dir, self.MODEL_BIN_FILENAME)

        return model_bin.exists() and (model_bin.stat().st_size > 0)

    def process_audio_wav(self) -> Generator[bytes]:
        with open(self.audio_filepath.get(), mode="rb") as fp:
            with io.BytesIO(fp.read()) as input_wav_io:
                with wave.open(input_wav_io, "rb") as input_wav_file:
                    while audio_bytes := input_wav_file.readframes(self.SAMPLES_PER_CHUNK):
                        yield audio_bytes

    def process_audio(self, stt_stream: Iterable[bytes]) -> Generator[ProcessedAudioChunk]:
        # Initialize with audio settings
        self.audio_processor_buffer = AudioBuffer(self.AUDIO_PROCESSOR_BYTES)
        stt_processed_stream: Iterable[ProcessedAudioChunk] | None = None
        auto_gain: int = int(self.auto_gain.get())
        noise_suppression_level: int = int(self.noise_suppression_level.get())

        if auto_gain > 0 or noise_suppression_level > 0:
            self.chunking_enabled = True

            from webrtc_noise_gain import AudioProcessor

            self.audio_processor = AudioProcessor(auto_gain, noise_suppression_level)

            logging.debug(f"Using Audio Processor with auto_gain: '{auto_gain}', noise_suppression_level: '{noise_suppression_level}'")

            # noise suppression/auto gain/volume
            stt_processed_stream = self.process_enhance_audio(stt_stream)
        else:
            # Volume multiplier only
            stt_processed_stream = self.process_volume_only(stt_stream)

        if stt_processed_stream is None:
            self.set_info_label(f"Audio processing failed.", color="red")
            raise ValueError(stt_processed_stream)

        return stt_processed_stream

    def process_volume_only(
        self,
        audio_stream: Iterable[bytes],
        sample_rate: int = SAMPLE_RATE,
        sample_width: int = SAMPLE_WIDTH,
    ) -> Generator[ProcessedAudioChunk, None]:
        """Apply volume transformation only (no audio enhancements) with optional chunking."""
        start_time: int = time.time()
        ms_per_sample: int = sample_rate // 1000
        ms_per_chunk: int = (self.AUDIO_PROCESSOR_SAMPLES // sample_width) // ms_per_sample
        timestamp_ms: int = 0
        volume_multiplier: float = float(self.volume_multiplier.get())
        
        logging.debug(f"Volume transformation with ms_per_sample: '{ms_per_sample}', ms_per_chunk: '{ms_per_chunk}', timestamp_ms: '{timestamp_ms}', volume_multiplier: '{volume_multiplier}'")

        for chunk in audio_stream:
            if volume_multiplier != 1.0:
                chunk = self.multiply_volume(chunk, volume_multiplier)

            if self.chunking_enabled:
                # 10 ms chunking
                for chunk_10ms in self.chunk_samples(
                    chunk, self.AUDIO_PROCESSOR_BYTES, self.audio_processor_buffer
                ):
                    yield ProcessedAudioChunk(
                        audio=chunk_10ms,
                        timestamp_ms=timestamp_ms,
                        is_speech=None,  # no VAD
                    )
                    timestamp_ms += ms_per_chunk
            else:
                # No chunking
                yield ProcessedAudioChunk(
                    audio=chunk,
                    timestamp_ms=timestamp_ms,
                    is_speech=None,  # no VAD
                )
                timestamp_ms += (len(chunk) // sample_width) // ms_per_sample

        logging.debug(f"Volume transformation done in {round(time.time() - start_time, self.ROUND_SECONDS)} seconds.")

    def process_enhance_audio(
        self,
        audio_stream: Iterable[bytes],
        sample_rate: int = SAMPLE_RATE,
        sample_width: int = SAMPLE_WIDTH,
    ) -> Generator[ProcessedAudioChunk, None]:
        """Split audio into 10 ms chunks and apply VAD/noise suppression/auto gain/volume transformation."""
        if self.audio_processor is None:
            self.set_info_label(f"Could not enhance audio. Invalid audio processor.", color="red")
            raise ValueError(self.audio_processor)

        start_time: int = time.time()
        ms_per_sample: int = sample_rate // 1000
        ms_per_chunk: int = (self.AUDIO_PROCESSOR_SAMPLES // sample_width) // ms_per_sample
        timestamp_ms: int = 0
        volume_multiplier: float = float(self.volume_multiplier.get())

        logging.debug(f"Enhance audio with ms_per_sample: '{ms_per_sample}', ms_per_chunk: '{ms_per_chunk}', timestamp_ms: '{timestamp_ms}', volume_multiplier: '{volume_multiplier}'")

        for dirty_samples in audio_stream:
            if volume_multiplier != 1.0:
                # Static gain
                dirty_samples = self.multiply_volume(dirty_samples, volume_multiplier)

            # Split into 10ms chunks for audio enhancements/VAD
            for dirty_10ms_chunk in self.chunk_samples(
                dirty_samples, self.AUDIO_PROCESSOR_BYTES, self.audio_processor_buffer
            ):
                ap_result = self.audio_processor.Process10ms(dirty_10ms_chunk)

                yield ProcessedAudioChunk(
                    audio=ap_result.audio,
                    timestamp_ms=timestamp_ms,
                    is_speech=ap_result.is_speech,
                )

                timestamp_ms += ms_per_chunk

        logging.debug(f"Enhance audio done in {round(time.time() - start_time, self.ROUND_SECONDS)} seconds.")

    def multiply_volume(self, chunk: bytes, volume_multiplier: float) -> bytes:
        """Multiplies 16-bit PCM samples by a constant."""

        def _clamp(val: float) -> float:
            """Clamp to signed 16-bit."""
            return max(-32768, min(32767, val))

        return array.array(
            "h",
            (int(_clamp(value * volume_multiplier)) for value in array.array("h", chunk)),
        ).tobytes()

    def chunk_samples(
        self,
        samples: bytes,
        bytes_per_chunk: int,
        leftover_chunk_buffer: AudioBuffer,
    ) -> Iterable[bytes]:
        """Yield fixed-sized chunks from samples, keeping leftover bytes from previous call(s)."""

        if (len(leftover_chunk_buffer) + len(samples)) < bytes_per_chunk:
            # Extend leftover chunk, but not enough samples to complete it
            leftover_chunk_buffer.append(samples)
            return

        next_chunk_idx: int = 0

        if leftover_chunk_buffer:
            # Add to leftover chunk from previous call(s).
            bytes_to_copy = bytes_per_chunk - len(leftover_chunk_buffer)
            leftover_chunk_buffer.append(samples[:bytes_to_copy])
            next_chunk_idx = bytes_to_copy

            # Process full chunk in buffer
            yield leftover_chunk_buffer.bytes()
            leftover_chunk_buffer.clear()

        while next_chunk_idx < len(samples) - bytes_per_chunk + 1:
            # Process full chunk
            yield samples[next_chunk_idx : next_chunk_idx + bytes_per_chunk]
            next_chunk_idx += bytes_per_chunk

        # Capture leftover chunks
        if rest_samples := samples[next_chunk_idx:]:
            leftover_chunk_buffer.append(rest_samples)


def restart_app(app: App):
    """Restarts the current app. Re-creates the entire environment."""
    logging.debug("Restarting...")
    app.destroy()
    App().mainloop()


##### BEGIN EXEC #####

if __name__ == "__main__":
    App().mainloop()
