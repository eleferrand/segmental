# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from os import path
from python_speech_features import delta
from python_speech_features import mfcc
from speech_dtw import qbe
from pydub import AudioSegment
import _pickle as pickle
import multiprocessing
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import sys
import os
import json
import io
import tgt
from progress.bar import Bar
from datetime import datetime
sys.path.append("..")
sys.path.append(path.join("..", "utils"))

def to_tuple(obj):
    sortie=[]
    for elt in obj:
        elt=tuple(elt)
        sortie.append(elt)
    return tuple(sortie)

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


def do_html(l, p):
    l = sorted(l, key=lambda x: (x[0], x[1]))
    with open("result.html", mode="w", encoding="utf8") as ficEcr:
        ficEcr.write("<html>\n\t<body>\n In the codes, u is for utterance, q for query and e for example\n<br/>\n")
        for i in l:
            pres=((p[i[0]]["TP"]/p[i[0]]["TOT"])*100)
            ficEcr.write("\t\t{} -> cost: {}, <a href=\"./data/{}.wav\">{}</a> precision : {}/{} = {}\n<br/>\n".format
                         (i[0], round(i[1], 3), i[2],i[2], p[i[0]]["TP"], p[i[0]]["TOT"], round(pres, 2)))
        ficEcr.write("\t</body>\n</html>")

def grab_data(root, path):
    # Grab spoken lexicon and return word, crop and duration of the word
    with io.open(root+path, mode="r",encoding="utf-8") as json_file:
        lex = json.load(json_file)
    search_mfcc_list = []
    for elt in lex:
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
        query["thres"] = 1
        search_mfcc_list.append(query)
    return(search_mfcc_list)


def grab_corp(root, corpus):
    # compare the queries to the utterances and return a dico with dtw value
    audios = []
    # Grabbing spoken utterances
    if os.path.isfile("./mfcc_corp.json"):
        print("Reading corpus mfccs")
        with open("./mfcc_corp.json", mode='r', encoding="utf8") as jfile:
            audios = json.load(jfile)
    else:
        for wav_fn in glob.glob(path.join(root + corpus, "*.wav")):
            print("Reading:", wav_fn)
            dic = {}
            dic["file"] = wav_fn
            dic["data"] = (get_mfcc_dd(wav_fn)).tolist()
            audios.append(dic)
        with io.open("./mfcc_corp.json", mode='w', encoding='utf8') as corp_json_file:
            json.dump(list(audios), corp_json_file)
    return audios

def crop_audio(elt):
    # time = elt["whole"].index(elt["cost"]) * 30
    time = elt["time"]*1000
    deb = time - 0.5
    if deb<0:
        deb=0
    fin = time + elt["duree"] +1
    ext = AudioSegment.from_wav(elt['ref'])
    ext = ext[deb:fin]
    ext.export("./data/{}.wav".format(elt["code"]), format("wav"))
    sortie=get_mfcc_dd("./data/{}.wav".format(elt["code"]))
    return sortie



class DTW():
    def __init__(self, queries, search, limit):
        self.limit = limit
        if self.limit:
            self.mode="thr"
        else:
            self.mode="no_thr"

        self.save_html=[]
        self.queries = queries
        self.search = search
        if os.path.isfile("/home/leferrae/Desktop/These/speech_dtw/processed/dtw_scores_{}.pkl".format(self.mode)):
            with open("/home/leferrae/Desktop/These/speech_dtw/processed/dtw_scores_{}.pkl".format(self.mode), mode="rb") as jfile:
                self.dtw_costs=pickle.load(jfile)
        else:
            self.dtw_costs = {}
        if os.path.isfile("/home/leferrae/Desktop/These/speech_dtw/processed/c_dict_{}.pkl".format(self.mode)):
            with open("/home/leferrae/Desktop/These/speech_dtw/processed/c_dict_{}.pkl".format(self.mode), mode="rb") as jfile:
                self.c_dict = pickle.load(jfile)
        else:
            self.c_dict = set()
        self.threshold = 1
        self.pres_mot = {}
        self.precision = {}
        self.precision["TP"] = 0
        self.precision["TOT"] = 0
        self.recall = 0
        self.par_mot = False

    def get_dtw(self):
        return self.dtw_costs


    def eval_lex(self, corp, size_queries):
        found=0
        tot=0
        queries = self.queries[0:size_queries]
        for query in range(0, len(queries)):
            mot = queries[query]["word"]
            ref = queries[query]["ref"][0]
            ind_quer = query
            for utt in self.search:
                if os.path.basename(utt["file"]) == ref:
                    ind_utt = self.search.index(utt)
            code=(ind_quer, ind_utt, 0)

            if code not in self.c_dict:
                self.precision["TP"] = self.precision["TP"] + 1
                self.precision["TOT"] = self.precision["TOT"] + 1
                for gold in sorted(glob.glob(path.join(corp, "*.mb.cleaned"))):
                    with open(gold, mode="r", encoding="utf8") as g_op:
                        if mot in g_op.read():
                            self.recall+=1

            self.c_dict.add(code)
        return tot

    def do_dtw(self, size_queries):
        """

        :param size_queries:
        :return:
        """
        print(datetime.now())

        bar = Bar("Processing dtw", max=len(self.search))
        for elt in range(0,len(self.search)):
            for query in range(0,len(self.queries[0:size_queries])):
                for inst in range(0,len(self.queries[query]["data"])):
                    code = (query, elt, inst)
                    if code not in self.c_dict:
                        ref = self.search[elt]["file"]
                        mot = self.queries[query]["word"]
                        query_mfcc = self.queries[query]["data"][inst]
                        search_mfcc = np.asarray(self.search[elt]["data"])
                        costs = qbe.dtw_sweep(query_mfcc, search_mfcc)
                        self.c_dict.add(code)
                        if mot not in self.dtw_costs:
                            self.dtw_costs[mot] = []
                        fin = {}
                        fin["duree"] = self.queries[query]["duree"]
                        fin["word"] = mot
                        fin["ref"] = ref
                        # fin["costs"] = costs
                        fin["cost"] = np.min(costs)
                        fin['time'] = costs.index(fin["cost"])*3/100
                        fin["code"] = "u{}q{}e{}".format(elt, query, inst)
                        fin["checked"] = False
                        self.dtw_costs[mot].append(fin)
            bar.next()
        bar.finish()

        with open("/home/leferrae/Desktop/These/speech_dtw/processed/dtw_scores_{}.pkl".format(self.mode), mode="wb") as jfile:
            pickle.dump(self.dtw_costs, jfile)
        with open("/home/leferrae/Desktop/These/speech_dtw/processed/c_dict.pkl_{}".format(self.mode), mode="wb") as jfile:
            pickle.dump(self.c_dict, jfile)
        print("dtw computed")
        self.par_mot=True
        print(datetime.now())

    def do_precision(self, quota, max_inst):
        """

        :param quota:
        :return:
        """
        print("computing precision")
        for mot in self.dtw_costs:
            self.dtw_costs[mot] = [x for x in self.dtw_costs[mot] if x["cost"] < self.threshold]
        for mot in self.dtw_costs:
            self.dtw_costs[mot] = sorted(self.dtw_costs[mot], key=lambda x: (x["cost"]))
        found = 0
        cost_temp = 0
        cpt_quota = 0
        q_mot=quota//len(self.dtw_costs)
        verif = {}#dict to check if each word is checked
        #pour chaque dtw score
        for mot in sorted(self.dtw_costs, key=lambda mot : len(self.dtw_costs[mot])):
            verif[mot] = 0
            found_m = 0
            #I check each word if it has not been checked according to q_mot and quota
            for elt in self.dtw_costs[mot]:
                # if quota > 0 :
                if verif[mot]<quota:
                    if elt["checked"] == False:
                        verif[mot]+=1
                        cpt_quota +=1
                        elt["checked"] = True
                        file_ref="/home/leferrae/Desktop/These/mboshi/wrd/"+(os.path.basename(elt["ref"])).replace(".wav", ".wrd")
                        # file_ref = elt["ref"].replace(".wav", ".mb.cleaned")
                        with open(file_ref, mode='r', encoding='utf-8') as op_ref:
                            gold = op_ref.read()
                        csn = True

                        if elt["word"] in gold:
                            wrd=gold.split("\n")
                            for line in wrd:
                                linesp=line.split()
                                if len(linesp)>2:

                                    if elt["time"]>float(linesp[0])-0.5 and elt["time"]<float(linesp[0])+0.5:
                                        if elt["word"]==linesp[2].lower():
                                            found += 1
                                            found_m +=1
                                            csn=False
                                            if elt["cost"] > cost_temp:
                                                cost_temp = elt["cost"]
                                            for inp in self.queries:
                                                if (elt['word'] in inp["word"]) and (elt['ref'] not in inp['ref']):
                                                    if len(inp['ref']) < max_inst:
                                                        inst = crop_audio(elt)
                                                        inp['data'].append(inst)
                                                        inp['ref'].append(elt['ref'])
                                                        self.save_html.append([elt["word"], elt["cost"], elt["code"]])


            if mot not in self.pres_mot:
                self.pres_mot[mot] = {}
                self.pres_mot[mot]["TP"] = found_m
                self.pres_mot[mot]["TOT"] = verif[mot]
            else:
                self.pres_mot[mot]["TP"] += found_m
                self.pres_mot[mot]["TOT"] += verif[mot]

        if self.limit:
            if self.threshold == 1:
                self.threshold = cost_temp

        if cpt_quota != 0:
            print((found / cpt_quota) * 100)
            print("{}/{}".format(found, cpt_quota))
        self.precision["TP"] = self.precision["TP"] + found
        self.precision["TOT"] = self.precision["TOT"] + cpt_quota

    def eval(self, size_queries, corp, root):

        self.eval_lex(corp, size_queries)
        tot = self.recall

        print("recall : {}  {}/{}\nprecision : {}  {}/{}".format((self.precision["TP"] / tot) * 100,
                                                           self.precision["TP"], tot,
                                                           (self.precision["TP"]/self.precision["TOT"]*100),
                                                                 self.precision["TP"], self.precision["TOT"]))
        if self.par_mot==True:
            for mot in self.pres_mot:
                if self.pres_mot[mot]["TOT"]==0:
                    print("precision {} = {}".format(mot, 0))
                else:
                    print("precision {} = {}".format(mot, (self.pres_mot[mot]["TP"]/self.pres_mot[mot]["TOT"])*100))
            with open("eval2708.txt", mode="w", encoding="utf8") as ficEcr:
                ficEcr.write("recall : {}\nprecision : {}\n".format((self.precision["TOT"] / self.recall) * 100, self.precision["TP"]/self.precision["TOT"]))
                for mot in self.pres_mot:
                    if self.pres_mot[mot]["TOT"]==0:
                        ficEcr.write(
                            "precision {} = {}\n".format(mot, 0))
                    else:
                        ficEcr.write("precision {} = {}\n".format(mot, (self.pres_mot[mot]["TP"] / self.pres_mot[mot]["TOT"]) * 100))
            do_html(self.save_html, self.pres_mot)
            self.par_mot=False