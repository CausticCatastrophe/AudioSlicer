import os
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import gradio as gr

# Utility functions
def open_output_directory():
    os.startfile(output_dir)


def GetTime(video_seconds):
    if (video_seconds < 0) :
        return 00
    else:
        sec = timedelta(seconds=float(video_seconds))
        d = datetime(1,1,1) + sec
        instant = str(d.hour).zfill(2) + ':' + str(d.minute).zfill(2) + ':' + str(d.second).zfill(2) + str('.001')
        return instant


def GetTotalTime(video_seconds):
    sec = timedelta(seconds=float(video_seconds))
    d = datetime(1,1,1) + sec
    delta = str(d.hour) + ':' + str(d.minute) + ":" + str(d.second)
    return delta


def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]


def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))


def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1


def convert_to_wav(input_file):
    # Check if the input file is already in WAV format
    if os.path.splitext(input_file)[-1].lower() == ".wav":
        return input_file

    # Convert the audio file to WAV format using pydub
    audio = AudioSegment.from_file(input_file)
    output_file = os.path.splitext(input_file)[0] + ".wav"
    audio.export(output_file, format="wav")

    return output_file


def read_wav(input_filename):
    return wavfile.read(filename=input_filename, mmap=True)


def calculate_window_energy(samples, sample_rate, window_duration, step_duration, max_energy):
    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)

    signal_windows = windows(
        signal=samples,
        window_size=window_size,
        step_size=step_size
    )

    window_energy = (energy(w) / max_energy for w in tqdm(
        signal_windows,
        total=int(len(samples) / float(step_size))
    ))

    return window_energy


def find_cut_samples(window_silence, sample_rate, step_duration):
    cut_times = (r * step_duration for r in rising_edges(window_silence))
    cut_samples = [int(t * sample_rate) for t in cut_times]
    cut_samples.append(-1)

    return cut_samples


def write_output_files(output_dir, output_filename_prefix, cut_samples, sample_rate, samples):
    cut_ranges = [(i, cut_samples[i], cut_samples[i+1]) for i in range(len(cut_samples) - 1)]

    video_sub = {str(i) : [str(GetTime(((cut_samples[i])/sample_rate))), 
                           str(GetTime(((cut_samples[i+1])/sample_rate)))] 
                 for i in range(len(cut_samples) - 1)}

    for i, start, stop in tqdm(cut_ranges):
        output_file_path = "{}_{:03d}.wav".format(
            os.path.join(output_dir, output_filename_prefix),
            i
        )
        print("Writing file {}".format(output_file_path))
        wavfile.write(
            filename=output_file_path,
            rate=sample_rate,
            data=samples[start:stop]
        )

    with open (output_dir+'\\'+output_filename_prefix+'.json', 'w') as output:
        json.dump(video_sub, output)


def split_audio(input_file, window_duration, silence_threshold):
    output_filename_prefix = os.path.splitext(os.path.basename(input_file))[0]

    input_filename = convert_to_wav(input_file)

    sample_rate, samples = read_wav(input_filename)
    max_amplitude = np.iinfo(samples.dtype).max
    max_energy = energy([max_amplitude])

    step_duration = window_duration / 10.

    window_energy = calculate_window_energy(samples, sample_rate, window_duration, step_duration, max_energy)
    window_silence = (e > silence_threshold for e in window_energy)

    cut_samples = find_cut_samples(window_silence, sample_rate, step_duration)
    write_output_files(output_dir, output_filename_prefix, cut_samples, sample_rate, samples)

    return output_dir


# Set your desired output directory here
script_path = os.path.abspath(__file__)
output_dir = os.path.join(os.path.dirname(script_path), "output")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

iface = gr.Interface(
    fn=split_audio,
    inputs=[
        gr.Audio(type="filepath"),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.6, label="Window Duration"),
        gr.Slider(minimum=0, maximum=0.5, step=1e-4, value=1e-4, label="Silence Threshold")
    ],
    outputs="text",
    title="Audio Splitter",
    description="Split audio files based on silence."
)

iface.launch()

