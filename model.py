from os import path
from datetime import datetime

import hashlib
import os
import data
import tensorflow as tf
import numpy as np
import batching
import training
import pickle
from tflego import NP_DTYPE, TF_DTYPE, NP_ITYPE, TF_ITYPE
import tflego
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist

from data import get_mfcc_dd
from pydub import AudioSegment
from progress.bar import Bar



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

class Auto_encoder():

    def __init__(self, queries, corpus, data_dir, train_tag, max_length, min_length, rnn_type, enc_n_hiddens,
                 dec_n_hiddens, n_z, learning_rate, keep_prob, ae_n_epochs, ae_batch_size, ae_n_buckets, cae_n_epochs,
                 cae_batch_size, rnd_seed=1, bidirectional=False):
        self.queries = queries
        self.corpus = corpus
        self.pair_list = []
        self.retrieved = {}
        self.pres_mot = {}
        self.precision = {}
        self.precision["TP"] = 0
        self.precision["TOT"] = 0
        self.save_html = []
        self.threshold = 0.1

        self.options_dict = {}
        self.options_dict["data_dir"] = data_dir
        self.options_dict["max_length"] = max_length
        self.options_dict["train_tag"] = train_tag
        self.options_dict["min_length"] = min_length
        self.options_dict["rnn_type"] = rnn_type
        self.options_dict["enc_n_hiddens"] = enc_n_hiddens
        self.options_dict["dec_n_hiddens"] = dec_n_hiddens
        self.options_dict["n_z"] = n_z
        self.options_dict["learning_rate"] = learning_rate
        self.options_dict["keep_prob"] = keep_prob
        self.options_dict["ae_n_epochs"] = ae_n_epochs
        self.options_dict["ae_batch_size"] = ae_batch_size
        self.options_dict["ae_n_buckets"] = ae_n_buckets
        self.options_dict["cae_n_epochs"] = cae_n_epochs
        self.options_dict["cae_batch_size"] = cae_batch_size
        self.options_dict["rnd_seed"] = rnd_seed
        self.options_dict["bidirectional"] = bidirectional

        hasher = hashlib.md5(repr(sorted(self.options_dict.items())).encode("ascii"))

        hash_str = hasher.hexdigest()[:10]
        self.model_dir = path.join(
            "models", path.split(self.options_dict["data_dir"])[-1] + "." +
                      self.options_dict["train_tag"], hash_str
        )
        self.options_dict_fn = path.join(self.model_dir, "self.options_dict.pkl")
        print("Model directory:", self.model_dir)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        print("Options:", self.options_dict)

        # Random seeds
        np.random.seed(self.options_dict["rnd_seed"])
        tf.set_random_seed(self.options_dict["rnd_seed"])

    def make_model(self):

        d_frame = 13  # None
        self.options_dict["n_input"] = d_frame

        self.pretrain_intermediate_model_fn = path.join(self.model_dir, "ae.tmp.ckpt")
        self.pretrain_model_fn = path.join(self.model_dir, "ae.best_val.ckpt")
        self.intermediate_model_fn = path.join(self.model_dir, "cae.tmp.ckpt")
        self.model_fn = path.join(self.model_dir, "cae.best_val.ckpt")
        # Model graph

        self.a = tf.placeholder(TF_DTYPE, [None, None, self.options_dict["n_input"]])
        self.a_lengths = tf.placeholder(TF_ITYPE, [None])
        self.b = tf.placeholder(TF_DTYPE, [None, None, self.options_dict["n_input"]])
        self.b_lengths = tf.placeholder(TF_ITYPE, [None])
        self.network_dict = self.build_cae_from_options_dict(
            self.a, self.a_lengths, self.b_lengths, self.options_dict
        )
        mask = self.network_dict["mask"]
        self.z = self.network_dict["z"]
        y = self.network_dict["y"]

        # Reconstruction loss
        self.loss = tf.reduce_mean(
            tf.reduce_sum(tf.reduce_mean(tf.square(self.b - y), -1), -1) /
            tf.reduce_sum(mask, 1))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.options_dict["learning_rate"]
        ).minimize(self.loss)


    def enc_queries(self, size_queries):

        queries = self.queries[: size_queries]
        train_tag = self.options_dict["train_tag"]
        min_length = None
        if self.options_dict["train_tag"] == "rnd":
            min_length = self.options_dict["min_length"]
            train_tag = "all"
        npz_fn = path.join(
            self.options_dict["data_dir"], "train." + train_tag + ".npz"
        )

        self.train_x, self.train_labels, train_lengths, train_keys = (
            data.load_data(queries))

        # Truncate and limit dimensionality
        max_length = self.options_dict["max_length"]
        d_frame = self.options_dict["n_input"]
        print("Limiting dimensionality:", d_frame)
        print("Limiting length:", max_length)
        data.trunc_and_limit_dim(x=self.train_x, lengths=train_lengths, d_frame=d_frame, max_length=max_length)

        # Get pairs

        self.pair_list.append([(i,i) for i in range(0, len(queries))])

        self.train_ae(self.train_x)

    def build_cae_from_options_dict(self, a, a_lengths, b_lengths, options_dict):

        # Latent layer
        build_latent_func = tflego.build_autoencoder
        latent_func_kwargs = {
            "enc_n_hiddens": [],
            "n_z": options_dict["n_z"],
            "dec_n_hiddens": [options_dict["dec_n_hiddens"][0]],
            "activation": tf.nn.relu
        }


        # Network
        network_dict = tflego.build_multi_encdec_lazydynamic_latentfunc(
            a, a_lengths, options_dict["enc_n_hiddens"],
            options_dict["dec_n_hiddens"], build_latent_func, latent_func_kwargs,
            y_lengths=b_lengths, rnn_type=options_dict["rnn_type"],
            bidirectional=options_dict["bidirectional"],
            keep_prob=options_dict["keep_prob"],
            add_conditioning_tensor=None )

        encoder_states = network_dict["encoder_states"]
        ae = network_dict["latent_layer"]
        z = ae["z"]
        y = network_dict["decoder_output"]
        mask = network_dict["mask"]
        y *= tf.expand_dims(mask, -1)  # safety

        return {"z": z, "y": y, "mask": mask}

    def train_ae(self, train_x):
        self.val_model_fn = self.pretrain_intermediate_model_fn
        train_batch_iterator = batching.PairedBucketIterator(
            train_x, [(i, i) for i in range(len(train_x))],
            self.options_dict["ae_batch_size"], self.options_dict["ae_n_buckets"],
            shuffle_every_epoch=True, speaker_ids=None )

        ae_record_dict = training.train_fixed_epochs_external_val(
            self.options_dict["ae_n_epochs"], self.optimizer, self.loss, train_batch_iterator,
            [self.a, self.a_lengths, self.b, self.b_lengths],
            save_model_fn=self.pretrain_intermediate_model_fn,
            save_best_val_model_fn=self.pretrain_model_fn
        )

        record_dict_fn = path.join(self.model_dir, "record_dict.pkl")
        print("Writing:", record_dict_fn)
        with open(record_dict_fn, "wb") as f:
            pickle.dump(ae_record_dict, f, -1)


        # Save options_dict
        options_dict_fn = path.join(self.model_dir, "options_dict.pkl")
        print("Writing:", options_dict_fn)
        with open(options_dict_fn, "wb") as f:
            pickle.dump(self.options_dict, f, -1)

    def segment(self, corp, step=3, wind=[100, 70, 40]):
        """

        :param corp: corpus from which the segment are extracted
        :param step:
        :param wind: the lenghts of the differents segments extracted at a same time
        :return: a stack of embeded segments and a reference file
        """
        segs = []
        seg_ref = {}
        cpt = 0
        for i, u in enumerate(corp):
            utt=u["data"]
            i_start = 0
            n_search = utt.shape[0]
            seg_list = []
            utt_lengths = []
            while i_start <= n_search - 1 or i_start == 0:
                for ind in wind:
                    sref = {}
                    sref["file"] = os.path.basename(u["file"])
                    sref["wind"] = ind
                    sref["time"] = i_start/100
                    seg_ref[cpt] = sref
                    seg=utt[i_start:i_start+ind]
                    seg_list.append(seg)
                    utt_lengths.append(seg.shape[0])
                    i_start += step
                    cpt +=1
            data.trunc_and_limit_dim(x=seg_list, lengths=utt_lengths, d_frame=self.options_dict["n_input"],
                                     max_length=self.options_dict["max_length"])
            batch_iterator = batching.SimpleIterator(seg_list, len(seg_list), False)
            saver = tf.train.Saver()
            with tf.Session() as session:
                saver.restore(session, self.val_model_fn)
                for batch_x_padded, batch_x_lengths in batch_iterator:
                    np_x = batch_x_padded
                    np_x_lengths = batch_x_lengths
                    np_z = session.run(
                        [self.z], feed_dict={self.a: np_x, self.a_lengths: np_x_lengths}
                    )[0]
                # break  # single batch
            segs.append(np_z)
        with open("./processed/segs.pkl", mode="wb") as pk:
            pickle.dump(np.vstack(segs), pk)
        with open("./processed/seg_ref", mode = "wb") as pk:
            pickle.dump(seg_ref, pk)
        return np.vstack(segs), seg_ref

    def do_dist(self, quota):
        iterator = batching.SimpleIterator(self.train_x, len(self.train_x), False)
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, self.val_model_fn)
            for batch_x_padded, batch_x_lengths in iterator:
                np_x = batch_x_padded
                np_x_lengths = batch_x_lengths
                emb_queries = session.run(
                    [self.z], feed_dict={self.a: np_x, self.a_lengths: np_x_lengths}
                )[0]
        embed_dict = {}

        for i, utt_key in enumerate(
                [self.train_labels[i] for i in iterator.indices]):
            embed_dict[utt_key] = emb_queries[i]

        # for z, w in enumerate(embed_dict):
        # self.retrieved[w] = []
        quers = np.vstack(emb_queries)
        print(datetime.now())
        print("segmentation")
        if os.path.isfile("./processed.segs.pkl"):
            with open("./processed/segs.pkl", mode="rb") as pk:
                segs = pickle.load(pk)
            with open("./processed/seg_ref.pkl", mode="rb") as pk:
                seg_ref = pickle.load(pk)
        else:
            segs, seg_ref = self.segment(self.corpus)
        print("distances")
        distances = cdist(segs, quers)
        ret = []
        for i, seg in enumerate(distances):
            mn = np.min(seg)
            w = np.where(seg == mn)
            ind = (w[0][0])
            ret.append([mn, self.queries[ind]["word"], i])
        meilleur = []
        verif = {}
        sort_ret = sorted(ret, key= lambda x : x[0])
        double = set()
        for j, m in enumerate(sort_ret):
            if m[1] not in verif:
                verif[m[1]] = 0
            elif verif[m[1]] < quota:
                code = (seg_ref[m[2]]["time"], m[1] ,seg_ref[m[2]]["file"])
                if code not in double:
                    out = {}
                    out["dist"] = m[0]
                    out["word"] = m[1]
                    out["ref"] = seg_ref[m[2]]["file"]
                    out["time"] = seg_ref[m[2]]["time"]
                    out["duree"] = seg_ref[m[2]]["wind"]
                    meilleur.append(out)
                    verif[m[1]] += 1
                    double.add(code)

        print(datetime.now())
        return meilleur

    def do_precision(self, max_inst, quota):
        meilleur = self.do_dist(quota)
        found = 0
        for can in meilleur:
            print(can["word"])
            print(can["ref"])
            input()
            file_ref = "/home/leferrae/Desktop/These/mboshi/wrd/" + (os.path.basename(can["ref"])).replace(".wav", ".wrd")
            with open(file_ref, mode='r', encoding='utf-8') as op_ref:
                gold = op_ref.read()

            if can["word"] in gold:

                wrd = gold.split("\n")
                for line in wrd:
                    linesp = line.split()
                    if len(linesp) > 2:
                        if can["time"] > float(linesp[0]) - 0.5 and can["time"] < float(linesp[0]) + 0.5:
                            print(linesp[2])
                            if can["word"] == linesp[2].lower():
                                found += 1
                                # found_m += 1
                                for inp in self.queries:
                                    if (can['word'] in inp["word"]) and (can['ref'] not in inp['ref']):
                                        if len(inp['ref']) < max_inst:
                                            inst = crop_audio(can)
                                            inp['data'].append(inst)
                                            inp['ref'].append(can['ref'])
                                            self.save_html.append([can["word"], can["cost"], can["code"]])
        print("precision : {}%".format(found/len(meilleur)*100))


        """
        bar = Bar("computing distances", max=len(embed_dict)*len(self.corpus))
        for z, w in enumerate(embed_dict):
            self.retrieved[w] = []
            for i, utt in enumerate(self.corpus):
                seg = self.segment(utt["data"])
                sub_seg = []
                for j, s in enumerate(seg):
                    dist = cosine(s, embed_dict[w])
                    
                    sub_seg.append(dist)
                best_sub = min(sub_seg)
                time = sub_seg.index(best_sub)//3
                fin = {}
                for i in self.queries:
                    if i["word"] == w:
                        fin["duree"] = i["duree"]
                fin["word"] = w
                fin["ref"] = utt["file"]
                fin["dist"] = best_sub
                fin['time'] = time
                fin["code"] = "u{}q{}".format(i, z)
                fin["checked"] = False
                self.retrieved[w].append(fin)
                bar.next()
        bar.finish()
        """

