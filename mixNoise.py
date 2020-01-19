import os
import subprocess
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def load_fromYoutube(youtubeID, dtype):
    """fetch video data and extract raw audio (pcm)
        play raw audio file the following command:
        >>> aplay -f S16_LE -c2 -r22050 mixed.raw
        >>> ffplay -f s16le -ac 1 -ar 44k mixed.raw
    """
    from pytube import YouTube
    mp4fileName = youtubeID + ".mp4"
    rawfileName = youtubeID + ".raw"
    if not os.path.exists(mp4fileName):
        print("\ndownloading video file ...")
        yt = YouTube("https://youtu.be/" + youtubeID)
        yt.streams.first().download(".")
        os.rename(yt.title+".mp4", mp4fileName)
    else:
        print("already exist: %s" % mp4fileName)
        if not os.path.exists(rawfileName):
            subprocess.call(["ffmpeg -i " + mp4fileName + " -vn -f s16le -acodec pcm_s16le " + rawfileName], shell=True, cwd=".")

    # load data
    data = np.memmap(rawfileName, dtype=dtype)

    return data

def cal_snr(signal, noise):
    return 20.0*math.log10(cal_rms(signal)/cal_rms(noise))

def cal_rms(signal):
    return np.sqrt(np.mean(np.square(signal.astype(np.float64)), axis=-1))

def cal_adjusted_rms(signal_rms, snr):
    noise_rms = signal_rms / 10**(float(snr) / 20)
    return noise_rms

def output_raw(fileName, signal, dtype, verbose=0):
    if verbose > 0:
        print("output file: %s" % fileName)
    with open(fileName, mode="wb") as fd:
        fd.write(bytearray(signal))

def mixNoise(signal, noise, snr, dtype, how="adjust_noise", verbose=0):
    """
        :param how: choose adjusting method
            "adjust_noise"- change noise amplitude such that
            the SNR of the signal and the adjusted noise is
            the target SNR (3rd argument)

            "-26dbov"- change both ampltitude of signal and
            noise
        :type how: string
    """
    # align length
    num_reps = int(len(signal) / len(noise) + 1)
    noise = np.tile(noise, reps=num_reps)[:len(signal)]

    signal_rms = cal_rms(signal)
    noise_rms = cal_rms(noise)

    adjusted_noise_rms = cal_adjusted_rms(signal_rms, snr)

    # mix noise
    adjusted_noise = (noise * (adjusted_noise_rms / noise_rms)).astype(dtype)
    mixed = signal + adjusted_noise
    mixed_rms = cal_rms(mixed)

    if how=="-26dbov":
        # under 26DB from overflow
        adjusted_mixed_rms = cal_adjusted_rms(np.iinfo(dtype).max, 26.0)
        adjusted_mixed = (mixed * (adjusted_mixed_rms / mixed_rms)).astype(dtype)
    else:
        # normalize
        if (mixed.max(axis = 0) > np.iinfo(dtype).max):
            mixed = mixed * (np.iinfo(dtype).max/mixed.max(axis = 0))
        adjusted_mixed = mixed

    if verbose > 0:
        print("=============================")
        print("signal rms: %.3f" % signal_rms)
        print("noise rms: %.3f -> %.3f" % (noise_rms, adjusted_noise_rms))
        print("mixed rms: %.3f -> %.3f" % (mixed_rms, adjusted_mixed_rms))
        print("SNR tgt: %.3f observed: %.3f -> %.3f " % (snr, cal_snr(signal, noise), cal_snr(signal, adjusted_noise)))
        print("=============================")

    return adjusted_mixed

def test_1():
    BGMID = "ewGbdVpEPVM"
    #ID = "afWTH9rv6zs"
    ID = "eIu0CXKekI4"
    #ID = "hsO5GPZt5UE"
    TYPE = np.int16

    signal = load_fromYoutube(youtubeID=ID, dtype=TYPE)
    noise = load_fromYoutube(youtubeID=BGMID, dtype=TYPE)

    mix = mixNoise(signal, noise, snr=0, how="-26dbov", dtype=TYPE, verbose=1)
    output_raw("mixed.raw", mix, dtype=TYPE, verbose=1)
