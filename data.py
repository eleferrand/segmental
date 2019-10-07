import os
from python_speech_features import delta
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import json
import numpy as np
import io
import glob
from os import path
import _pickle as pickle

def grab_data(root, path):
    # Grab spoken lexicon and return word, crop and duration of the word
    with io.open(root+path, mode="r",encoding="utf-8") as json_file:
        lex = json.load(json_file)
    search_mfcc_list = []
    for i, elt in enumerate(lex):
        wav_fn = elt["crop"]
        rate, signal = wav.read(wav_fn)
        dur = len(signal) / rate * 1000
        query_mfcc = get_mfcc_dd(wav_fn)
        query = {}
        query["duree"] = dur
        query["data"] = []
        query["data"].append(query_mfcc)
        query["word"] = elt["mboshi"]
        query["ref"] = []
        query["ref"].append(elt["ref"])
        search_mfcc_list.append(query)
    return(search_mfcc_list)


def grab_corp(root, corpus):
    # compare the queries to the utterances and return a dico with dtw value
    audios = []
    # Grabbing spoken utterances
    if os.path.isfile("./mfcc_corp.pkl"):
        print("Reading corpus mfccs")
        with open("./mfcc_corp.pkl", mode='rb') as jfile:
            audios = pickle.load(jfile)
    else:
        for wav_fn in glob.glob(path.join(root + corpus, "*.wav")):
            print("Reading:", wav_fn)
            dic = {}
            dic["file"] = wav_fn
            dic["data"] = get_mfcc_dd(wav_fn)
            audios.append(dic)
        with io.open("./mfcc_corp.pkl", mode='wb') as corp_pkl_file:
            pickle.dump(audios, corp_pkl_file)
    return audios

def get_mfcc_dd(wav_fn, cmvn=True):
    """Return the MFCCs with deltas and delta-deltas for a audio file."""
    (rate, signal) = wav.read(wav_fn)
    mfcc_static = mfcc(signal, rate)
    mfcc_deltas = delta(mfcc_static, 2)
    mfcc_delta_deltas = delta(mfcc_deltas, 2)
    features = np.hstack([mfcc_static, mfcc_deltas, mfcc_delta_deltas])
    if cmvn:
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    return features

def load_data(queries):

    out = {}
    x = []
    keys = []
    labels = []
    train_lengths = []
    for i, query in enumerate(queries):
        feat = query["data"][0]
        key = i
        label = query["word"]
        train_length = feat.shape[0]
        x.append(feat)
        keys.append(key)
        labels.append(label)
        out[label] = feat
        train_lengths.append(train_length)
    return x, labels, train_lengths, keys

def trunc_and_limit_dim(x, lengths, d_frame, max_length):
    for i, seq in enumerate(x):
        x[i] = x[i][:max_length, :d_frame]
        lengths[i] = min(lengths[i], max_length)
