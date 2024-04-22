from typing import Any
from torch import nn as nn
import torchaudio
import random
import pyroomacoustics as pra
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile
import os
from audiomentations import AddGaussianSNR, Mp3Compression
# https://github.com/mattpitkin/pyroomacoustics/commit/a16bd1c571e961fe9a877774c16b6213f5c15e08


class NoiseAugmentation:
    def __init__(self, noise_root="/home/fred/Projetos/DATASETS/TAU-urban-acoustic-scenes-2022-mobile-development/audio/",
                 sr=24000,
                 background_noise_snr_max=30.0, background_noise_snr_min=5.0,
                 background_noise_probability=0.3, gaussian_noise_snr_range=(0, 30),
                 gaussian_noise_probability=0.3, reverb_conditions_probability=0.3,
                 reverb_conditions_reverbation_times=(0.2, 0.5),
                 reverb_conditions_room_dimensions={"xy": (2.0, 30.0), "z": (2.0, 15.0)},
                 reverb_conditions_room_params={"fs": 24000, "max_order": 10, "absorption": 0.2},
                 reverb_conditions_source_position=(1.0, 1.0, 1.0),
                 reverb_conditions_mic_position=(1.0, 0.7, 1.2), n_rirs=10,
                 mp3_compression_rate_range=(8, 16), mp3_compression_probability=0.3) -> None:
        
        self.sr = sr
        
        # background_noise
        self.background_noise_snr_max = background_noise_snr_max
        self.background_noise_snr_min = background_noise_snr_min

        self.background_noise_p = background_noise_probability
        self.noise_audio_paths = []
        self.noise_root = noise_root

        # gaussian_noise
        self.min_gaussian_snr = gaussian_noise_snr_range[0]
        self.max_gaussian_snr = gaussian_noise_snr_range[1]
        self.gaussian_noise_p = gaussian_noise_probability       
        self.GaussianNoiseFn = AddGaussianSNR(
                min_snr_db=self.min_gaussian_snr, 
                max_snr_db=self.max_gaussian_snr, 
                p=self.gaussian_noise_p
            )
        # reverb_conditions
        self.reverb_conditions_p = reverb_conditions_probability
        self.reverb_conditions_reverbation_times_max = reverb_conditions_reverbation_times[1]
        self.reverb_conditions_reverbation_times_min = reverb_conditions_reverbation_times[0]
        self.reverb_conditions_room_xy = {"max" : reverb_conditions_room_dimensions["xy"][1], "min" : reverb_conditions_room_dimensions["xy"][0]}
        self.reverb_conditions_room_z = {"max" : reverb_conditions_room_dimensions["z"][1], "min" : reverb_conditions_room_dimensions["z"][0]}
        self.reverb_conditions_room_params = {
            'fs' : self.sr, 
            'max_order': reverb_conditions_room_params["max_order"],
            'absorption': reverb_conditions_room_params["absorption"]
        }
        self.reverb_conditions_source_pos = reverb_conditions_source_position
        self.reverb_conditions_mic_pos = reverb_conditions_mic_position
        self.n_rirs = n_rirs # Default 1000
        self.rirs = []
        self.prepareRir(self.n_rirs)        
        
        # mp3_compression
        self.mp3_min_bitrate = mp3_compression_rate_range[0]
        self.mp3_max_bitrate = mp3_compression_rate_range[1]
        self.mp3_prob = mp3_compression_probability

        self.noise_audio_paths.extend(self.searchNoiseFiles(self.noise_root))
        if self.noise_audio_paths:  # If paths were found
            print(f"Found {len(self.noise_audio_paths)} noise background files.")
            
        else:
            print("No noise files were found.")   

    def searchNoiseFiles(self, noise_root):
        # Load noise audio paths
        path_list = list(Path(noise_root).rglob("*.wav"))  # Use rglob for recursive globbing
        return path_list

    def applyGaussianNoise(self, waveform):
        waveform = self.GaussianNoiseFn(waveform.numpy(), sample_rate=self.sr)
        return torch.tensor(waveform)
    
    def applyCodec(self, waveform):
        codec = Mp3Compression(min_bitrate=self.mp3_min_bitrate, max_bitrate=self.mp3_max_bitrate, p=self.mp3_prob)
        waveform = codec(waveform.numpy(), self.sr)
        return torch.tensor(waveform)

    def applyReverb(self, waveform):
        if len(self.rirs) == 0:
            raise RuntimeError
        rir = random.choice(self.rirs)
        augmented = torchaudio.functional.fftconvolve(waveform, rir)
        return augmented.float()

    def prepareRir(self, n_rirs):
        print("Calculating RIR values...")
        for i in tqdm(range(n_rirs)):
            xy_minmax = self.reverb_conditions_room_xy
            z_minmax = self.reverb_conditions_room_z
            x = random.uniform(xy_minmax['min'], xy_minmax['max'])
            y = random.uniform(xy_minmax['min'], xy_minmax['max'])
            z = random.uniform(z_minmax['min'], z_minmax['max'])
            corners = np.array([[0, 0], [0, y], [x, y], [x, 0]]).T
            room = pra.Room.from_corners(corners, **self.reverb_conditions_room_params)
            room.extrude(z)
            room.add_source(self.reverb_conditions_source_pos)
            room.add_microphone(self.reverb_conditions_mic_pos)

            room.compute_rir()
            rir = torch.tensor(np.array(room.rir[0]))
            rir = rir / rir.norm(p=2)
            self.rirs.append(rir)

    def applyBackgroundNoise(self, waveform):
        snr_max, snr_min = self.background_noise_snr_max, self.background_noise_snr_min
        snr = random.uniform(snr_min, snr_max)
        #print(f"Number of noise paths available: {len(self.noise_audio_paths)}")
        if len(self.noise_audio_paths) == 0:
            print("Warning: No noise audio paths found. Please check the configuration and path loading.")
        else:
            sample_paths = self.noise_audio_paths[:5]  # Adjust the number of paths as needed
            #print(f"Sample available noise paths: {sample_paths}")
        noise_path = random.choice(self.noise_audio_paths)
        #print(f"Applying noise from: {noise_path}") 
        noise, noise_sr = torchaudio.load(noise_path)
        noise /= noise.norm(p=2)
        if noise.size(0) > 1:
            noise = noise[0].unsqueeze(0)
        noise = torchaudio.functional.resample(noise, noise_sr, self.sr)
        if not noise.size(1) < waveform.size(1):
            start_idx = random.randint(0, noise.size(1) - waveform.size(1))
            end_idx = start_idx + waveform.size(1)
            noise = noise[:, start_idx:end_idx]
        else:
            noise = noise.repeat(1, waveform.size(1) // noise.size(1) + 1)[
                :, : waveform.size(1)
            ]
        augmented = torchaudio.functional.add_noise(
            waveform=waveform, noise=noise, snr=torch.tensor([snr])
        )
        return augmented

    def augment(self, waveform):
        wav_lenght = waveform.size(1)  
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)    
        if random.random() > self.background_noise_p:   
            waveform = self.applyBackgroundNoise(waveform)
            
        else:
            waveform = self.applyGaussianNoise(waveform) 
        
        if random.random() > self.reverb_conditions_p:
            waveform = self.applyReverb(waveform)
        # Condition not necessary, the function already has a probability parameter
        waveform = self.applyCodec(waveform)
        waveform = waveform[:,:wav_lenght]
        return waveform.squeeze()


if __name__ == "__main__":
    import os
    import torch
    import torchaudio
    from glob import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/home/fred/Projetos/DATASETS/BRSpeech_CML_TTS_v04012024/test/audio/3050/2941/")
    parser.add_argument("--output_dir", default="results_aug1")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    augmenter = NoiseAugmentation(sr=48000)

    for audio_filepath in tqdm(glob(args.input_dir + "/*.flac")):
        audio, sr = torchaudio.load(audio_filepath)

        aug_audio = augmenter.augment(audio.squeeze())

        filename = os.path.basename(audio_filepath)
        out_filepath = os.path.join(args.output_dir, filename)
        aug_audio_tensor = torch.tensor(aug_audio).unsqueeze(0)
        torchaudio.save(out_filepath.replace(".flac", "_augmented.wav"), aug_audio_tensor, sample_rate=48000)
