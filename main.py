from data import grab_data
from data import grab_corp
from model import Auto_encoder as AE
from os import path

iteration = 1
quota = 5
max_inst = 5

#serveur
# root="/home/getalp/leferrae/thesis"
# lexicon = "/corpora/crops/lex20/spoken_lex100.json"
# corpus = "/corpora/mboshi-french-parallel-corpus/full_corpus_newsplit/all/"
# file_ref = root + "/corpora/"

# #local
root = "/home/leferrae/Desktop/These"
lexicon = "/mboshi/crops/lex100/spoken_lex100.json"
corpus = "/mboshi/mboshi-french-parallel-corpus/full_corpus_newsplit/all"
file_ref = root + "/mboshi/"

queries = grab_data(root, lexicon)
data = grab_corp(root, corpus)[0:50]

model = AE(queries=queries, corpus=data,
           data_dir= path.join("data", "first"),
           train_tag="utd",
           max_length=100, min_length=50,
           rnn_type="gru", enc_n_hiddens=[400, 400, 400], dec_n_hiddens=[400, 400, 400],
           n_z=130, learning_rate=0.001, keep_prob=1.0, ae_n_epochs= 2, ae_batch_size= 4, ae_n_buckets= 3,
           cae_n_epochs=2, cae_batch_size=4, bidirectional=False)
model.make_model()
for i in range(0,iteration):
    print("iteration {}".format(i+1))
    model.enc_queries(size_queries=20*(i+1))
    model.do_precision(quota=quota, max_inst=max_inst)

