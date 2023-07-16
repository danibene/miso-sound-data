import os
import io
import pydub
import requests
import wget
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import shutil
from typing import Callable
from warnings import warn
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile


def is_url(x):
    """Check if a string is a URL"""
    # https://stackoverflow.com/questions/7160737/how-to-validate-a-url-in-python-malformed-or-not
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


def is_available(x):
    """Check if a file is available given URL or path"""
    if is_url(x):
        r = requests.head(x)
        return r.status_code == 200
    else:
        return Path(x).is_file()


def download_from_url(in_path, out_dir_path):
    """Download a file from a URL to a specified directory"""
    if is_url(in_path):
        # if output path does not exist, create it
        Path(out_dir_path).mkdir(parents=True, exist_ok=True)
    return wget.download(in_path, out=out_dir_path)


def load_audio(in_path):
    """Load audio without writing file"""
    if is_url(in_path):
        p = io.BytesIO()
        # https://stackoverflow.com/questions/59426275/download-and-open-file-with-librosa-without-writing-to-filesystem
        with urlopen(in_path) as r:
            r.seek = lambda *args: None  # allow pydub to call seek(0)
            pydub.AudioSegment.from_file(r).export(
                p, Path(in_path).suffix.split(".")[1]
            )
        p.seek(0)
    else:
        p = in_path
    return librosa.load(p, mono=True, sr=None)


def librosa_to_pydub(y, sr):
    # https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa
    # convert from float to uint16
    if y.dtype == np.dtype("float32"):
        y = np.array(y * (1 << 15), dtype=np.int16)
    audio_segment = pydub.AudioSegment(
        y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=1
    )
    return audio_segment


def pydub_to_librosa(audio_segment):
    # https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa
    channel_sounds = audio_segment.split_to_mono()
    sr = audio_segment.frame_rate
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    y = fp_arr.reshape(-1)
    return y, sr


def match_target_amplitude_pydub(y, sr=44100, level=-18.0):
    # Normalize with target level specified in dBFS
    # https://github.com/jiaaro/pydub/issues/90
    sound = librosa_to_pydub(y, sr)
    change_in_dBFS = level - sound.dBFS
    y, sr = pydub_to_librosa(sound.apply_gain(change_in_dBFS))
    return y


def normalize(y, sr=44100, level=-23.0, method="ffmpeg_normalize", **kwargs):
    if level is not None and method is not None:
        if method == "ffmpeg_normalize":
            try:
                import ffmpeg_normalize
            except:
                warn(
                    "Error importing ffmpeg_normalize. To install, run `pip install ffmpeg_normalize`"
                )
            tmp_pre_path = str(Path("tmp", "tmp_pre.wav"))
            tmp_post_path = str(Path("tmp", "tmp_post.wav"))
            normalizer = ffmpeg_normalize.FFmpegNormalize(
                sample_rate=sr, target_level=level, **kwargs
            )
            save_audio(out_path=tmp_pre_path, y=y, sr=sr)
            normalizer.add_media_file(tmp_pre_path, tmp_post_path)
            normalizer.run_normalization()
            y, sr = load_audio(in_path=tmp_post_path)
            os.remove(tmp_pre_path)
            os.remove(tmp_post_path)
            return y
        else:
            return match_target_amplitude_pydub(y, sr=sr, **kwargs)
    else:
        return y


def apply_fade(y, sr=44100, duration=0.010, inout="both"):
    """Apply fade in and out to a signal"""
    if duration > 0:
        # https://stackoverflow.com/questions/64894809/is-there-a-way-to-make-fade-out-by-librosa-or-another-on-python/65048786#65048786
        # convert duration to samples
        length = int(duration * sr)
        if inout == "both":
            inout_list = ["in", "out"]
        else:
            inout_list = [inout]
        for inout in inout_list:
            if inout == "out":
                end = y.shape[0]
                start = end - length
            else:
                start = 0
                end = start + length

            # compute fade out curve
            # linear fade
            fade_curve = np.linspace(1.0, 0.0, length)
            # apply the curve
            y[start:end] = y[start:end] * fade_curve
    return y


def segment_audio(y, sr=None, segment=None):
    """Segment an audio file"""
    # check if segment is string, then assume it is a path to a table file
    if isinstance(segment, str):
        segment = list(pd.read_table(segment, header=None).values[0][:2])
    # otherwise assume segment is list of 2 values
    if segment is not None:
        offset = segment[0]
        duration = segment[1] - segment[0]
    else:
        offset = 0.0
        duration = None

    # check if y is file name
    if isinstance(y, str):
        y, sr = librosa.load(y, mono=True, sr=sr, offset=offset, duration=duration)
    else:
        if duration is None:
            y = y[int(np.rint(sr * offset)) :]
        else:
            y = y[int(np.rint(sr * offset)) : int(np.rint(sr * (duration - offset)))]

    return y, sr


def process_audio(
    y, sr=44100, target_sr=44100, norm_level=None, fade_duration=0.010, **kwargs
):
    """Resample, then apply fade-in and fade-out. norm_level can be set to apply normalization."""
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return (
        apply_fade(
            normalize(y, sr=sr, level=norm_level), sr=sr, duration=fade_duration
        ),
        sr,
    )

def extract_zip(input_path: str, output_dir: str) -> None:
    """
    Extracts the contents of a ZIP archive to the specified directory.

    Args:
        input_path (str): Path to the input ZIP archive.
        output_dir (str): Path to the output directory for extracting the ZIP archive.

    Returns:
        None
    """
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


def convert_wav_to_flac(wav_path: str, flac_path: str) -> None:
    """
    Converts a WAV file to FLAC format.

    Args:
        wav_path (str): Path to the input WAV file.
        flac_path (str): Path to save the output FLAC file.

    Returns:
        None
    """
    wav_data, sr = librosa.load(wav_path, sr=None)
    sf.write(flac_path, wav_data, sr, format='flac')


def convert_files_in_directory(input_dir: str, output_dir: str, file_format: str, conversion_func: Callable) -> List[str]:
    """
    Converts files with the specified format in a directory using the given conversion function.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory for saving the converted files.
        file_format (str): File format of the files to be converted.
        conversion_func (callable): Function for converting the files to the desired format.

    Returns:
        List[str]: List of converted file paths.
    """
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    converted_files = []

    for file_path in input_dir_path.glob(f'**/*{file_format}'):
        rel_path = file_path.relative_to(input_dir_path)
        output_path = output_dir_path / rel_path.with_suffix('.flac')
        conversion_func(str(file_path), str(output_path))
        converted_files.append(str(output_path))

    return converted_files


def create_zip(input_dir: str, output_path: str) -> None:
    """
    Creates a ZIP archive from the contents of a directory.

    Args:
        input_dir (str): Path to the input directory.
        output_path (str): Path to the output ZIP archive.

    Returns:
        None
    """
    input_dir_path = Path(input_dir)

    with zipfile.ZipFile(output_path, 'w') as zip_output:
        for file_path in input_dir_path.glob('**/*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(input_dir_path)
                zip_output.write(file_path, arcname=str(rel_path))


def convert_wav_zip_to_flac_zip(input_zip: str, output_zip: str) -> None:
    """
    Converts WAV files in a ZIP archive to FLAC format and saves them in a new ZIP archive.

    Args:
        input_zip (str): Path to the input ZIP archive containing WAV files.
        output_zip (str): Path to the output ZIP archive for saving the converted FLAC files.

    Returns:
        None
    """
    temp_dir = "temp_conversion"

    # Extract the input ZIP file
    extract_zip(input_zip, temp_dir)

    # Convert WAV files to FLAC
    converted_files = convert_files_in_directory(temp_dir, temp_dir, ".wav", convert_wav_to_flac)

    # Create a new ZIP file with the converted FLAC files
    create_zip(temp_dir, output_zip)

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

def save_audio(out_path, y, sr):
    # if output parent path does not exist, create it
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, sr)


class MisoSoundLoader:
    def __init__(
        self,
        ids=None,
        root_out_path=str(Path("miso_sound_download")),
        segment_in_dir_path="https://raw.githubusercontent.com/miso-sound/miso-sound-annotate/main/segmentation",
        label_in_dir_path="https://raw.githubusercontent.com/miso-sound/miso-sound-annotate/main/labels",
        original_audio_in_dir_path="https://zenodo.org/api/records/7106450",
    ):
        """Loader for sound bank"""
        self.root_out_path = root_out_path
        self.ids = ids
        self.segment_in_dir_path = segment_in_dir_path
        self.label_in_dir_path = label_in_dir_path
        self.original_audio_in_dir_path = original_audio_in_dir_path

    def get_paths(
        self,
        original_audio_name_end="_original",
        processed_audio_name_end="_processed",
        original_audio_out_dir_name="original_audio",
        processed_audio_out_dir_name="processed_audio",
    ):
        """Get paths of input and output files"""
        ids = self.ids
        root_out_path = self.root_out_path
        segment_in_dir_path = self.segment_in_dir_path
        label_in_dir_path = self.label_in_dir_path
        label_out_dir_path = str(Path(root_out_path, "labels"))
        original_audio_in_dir_path = self.original_audio_in_dir_path
        original_audio_out_dir_path = str(
            Path(root_out_path, original_audio_out_dir_name)
        )
        processed_audio_out_dir_path = str(
            Path(root_out_path, processed_audio_out_dir_name)
        )
        all_paths = {}
        audio_paths = {}
        label_paths = {}
        original_audio_d = requests.get(original_audio_in_dir_path).json()
        audio_f_list = []
        if ids is None:
            original_audio_download_urls = []
            for f in original_audio_d["files"]:
                if original_audio_name_end in f["key"] and ".csv" not in f["key"]:
                    f["id"] = f["key"].split(original_audio_name_end)[0]
                    audio_f_list.append(f)
        else:
            for id in ids:
                for f in original_audio_d["files"]:
                    if str(id) + original_audio_name_end in f["key"]:
                        f["id"] = f["key"].split(original_audio_name_end)[0]
                        audio_f_list.append(f)
        for f in audio_f_list:
            id = f["id"]
            audio_paths[id] = {
                "original_audio_in_file_path": f["links"]["self"],
                "original_audio_out_dir_path": original_audio_out_dir_path,
                "segment_in_file_path": segment_in_dir_path
                + r"/"
                + str(id)
                + "_segment.txt",
                "processed_audio_out_file_path": str(
                    Path(
                        processed_audio_out_dir_path,
                        str(id) + processed_audio_name_end + ".wav",
                    )
                ),
            }
            label_paths[id] = {
                "label_out_dir_path": label_out_dir_path,
                "label_in_file_path": label_in_dir_path
                + r"/"
                + str(id)
                + "_labels.txt",
            }
        info_paths = {
            "original_metadata_in_file_path": [
                f for f in original_audio_d["files"] if ".csv" in f["key"]
            ][0]["links"]["self"],
            "original_metadata_out_dir_path": original_audio_out_dir_path,
        }
        all_paths["info"] = info_paths
        all_paths["audio"] = audio_paths
        all_paths["label"] = label_paths
        return all_paths

    def get_audio(
        self,
        save_original=True,
        save_processed=True,
        return_audio=True,
        segment_processed=True,
        process_func=process_audio,
        **kwargs
    ):
        """Load, segment and process audio"""
        audio_paths = self.get_paths()["audio"]
        audio = []
        for id, paths in audio_paths.items():
            if save_original:
                y = download_from_url(
                    in_path=paths["original_audio_in_file_path"],
                    out_dir_path=paths["original_audio_out_dir_path"],
                )
                sr = None
            else:
                y, sr = load_audio(in_path=paths["original_audio_in_file_path"])
            processed_available = True
            if segment_processed:
                if is_available(paths["segment_in_file_path"]):
                    y, sr = segment_audio(
                        y=y, sr=sr, segment=paths["segment_in_file_path"]
                    )
                else:
                    processed_available = False
            else:
                y, sr = segment_audio(y=y, sr=sr, segment=None)
            if processed_available:
                if process_func is not None:
                    y, sr = process_func(y, sr=sr, **kwargs)
                if save_processed:
                    save_audio(paths["processed_audio_out_file_path"], y, sr)
            if return_audio:
                audio.append({"sig": y, "sampling_rate": sr, "id": id})
        if return_audio:
            return audio

    def get_info(self, save=True):
        """Load metadata"""
        info_paths = self.get_paths()["info"]
        info_df = pd.read_csv(info_paths["original_metadata_in_file_path"])
        if save:
            download_from_url(
                in_path=info_paths["original_metadata_in_file_path"],
                out_dir_path=info_paths["original_metadata_out_dir_path"],
            )
        return info_df

    def get_labels(self, save=True):
        """Load audio labels"""
        labels = []
        label_paths = self.get_paths()["label"]
        for id, paths in label_paths.items():
            label = pd.read_table(paths["label_in_file_path"], header=None)
            label["id"] = id
            labels.append(label)
            if save:
                download_from_url(
                    in_path=paths["label_in_file_path"],
                    out_dir_path=paths["label_out_dir_path"],
                )
        label_df = pd.concat(labels)
        label_df.columns = ["start", "stop", "full_label", "id"]
        salience_d = {"C1": "foreground", "C2": "background"}
        label_df["salience"] = [
            salience_d[v.split("-")[0]] for v in label_df["full_label"].values
        ]
        label_df["label"] = [v.split("-")[1] for v in label_df["full_label"].values]
        return label_df.loc[:, ["id", "start", "stop", "label", "salience"]]
