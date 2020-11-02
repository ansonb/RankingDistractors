# coding=utf-8

import sys
import os
import numpy as np
import pickle
import time
import math
import datetime
import tensorflow as tf
from functools import wraps

import Discriminator
import Generator

from data_helpers import encode_sent, encode_sent_lstm
import data_helpers as data_helpers

from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertModel
from torch import nn, optim
import torch
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize

import re

# import dataHelper
# Data
tf.flags.DEFINE_string("dataset", "semevalQA", "dataset path")
tf.flags.DEFINE_string("prefix", "semevalQA", "prefix")
# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length_q", 100, "Max sequence length fo sentence (default: 100)")
tf.flags.DEFINE_integer("max_sequence_length_a", 28, "Max sequence length fo sentence (default: 100)")
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-6, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.0001, "learning_rate (default: 0.1)")
tf.flags.DEFINE_string("padding", "<a>", "dataset path")

# Training parameters
tf.flags.DEFINE_string("pretrained_embeddings_path", "", "path to pretrained_embeddings")
tf.flags.DEFINE_string("pretrained_model_path", "", "path to pretrained_model")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("pools_size", 64, "The positive sample set, which is bigger than 500")
tf.flags.DEFINE_integer("gan_k", 16, "the number of samples of gan")
tf.flags.DEFINE_integer("g_epochs_num", 1, " the num_epochs of generator per epoch")
tf.flags.DEFINE_integer("d_epochs_num", 1, " the num_epochs of discriminator per epoch")
tf.flags.DEFINE_integer("sampled_temperature", 5, " the temperature of sampling")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_integer("max_sent_len", 32, "Max sentence length")
tf.flags.DEFINE_integer("conv_filter_size", 3, "kernel size for char conv")
tf.flags.DEFINE_integer("max_word_len", 10, "Max word len for char conv")
tf.flags.DEFINE_integer("input_dim", 100, "input dim of the lstm layer")
tf.flags.DEFINE_integer("hidden_dim", 100, "hidden dim of the lstm layer")
tf.flags.DEFINE_integer("layers", 1, "Number of lstm layers")
tf.flags.DEFINE_boolean("is_bidirectional", True, "whether to use bidirectional lstm")
tf.flags.DEFINE_integer("char_embed_dim", 50, "dimension of the char embedding")
tf.flags.DEFINE_integer("char_feature_size", 50, "Output dimension of the char conv")
tf.flags.DEFINE_float("drop_out_rate", 0.0, "Drop rate")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
assert(FLAGS.batch_size == FLAGS.pools_size)

MAX_SENT_LEN = 32
BINARY_CLASSIFICATION = False

print(("\nParameters:"))
for attr, value in sorted(FLAGS.__flags.items()):
        print(("{}={}".format(attr.upper(), value)))
print((""))

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
# device = "cpu"
print(device)

print(("Loading data..."))
vocab, embd = data_helpers.build_vocab(FLAGS.dataset, FLAGS.pretrained_embeddings_path)
char_vocab = data_helpers.build_char_vocab(FLAGS.dataset)
if len(FLAGS.pretrained_embeddings_path) > 0:
    assert(embd.shape[1] == FLAGS.embedding_dim)
    with open('{}/embd.pkl'.format(FLAGS.dataset), 'wb') as fout:
        pickle.dump(embd, fout)
with open('{}/vocab.pkl'.format(FLAGS.dataset), 'wb') as fout:
    pickle.dump(vocab, fout)
with open('{}/char_vocab.pkl'.format(FLAGS.dataset), 'wb') as fout:
    pickle.dump(char_vocab, fout)
alist = data_helpers.read_alist_standalone(FLAGS.dataset, "vocab.txt", FLAGS.max_sequence_length_a, FLAGS.padding)
raw, raw_dict = data_helpers.read_raw_bert(FLAGS.dataset)
devList = data_helpers.loadTestSet(FLAGS.dataset, "valid.data")
testList = data_helpers.loadTestSet(FLAGS.dataset, "test.data")
testallList = data_helpers.loadTestSet(FLAGS.dataset, "test.data")  # testall
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

print("Load done...")
if not os.path.exists('./log/'):
    os.mkdir('./log/')
log_precision = 'log/{}.test.gan_precision.{}.log'.format(FLAGS.prefix, timeStamp)
log_loss = 'log/{}.test.gan_loss.{}.log'.format(FLAGS.prefix, timeStamp)


def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%s runed %.2f seconds" % (func.__name__, delta))
        return ret
    return _deco


def generate_gan(sess, model, negative_size=FLAGS.gan_k):
    '''used for generate negative samples for the Discriminator'''
    samples = []
    for _index, pair in enumerate(raw):
        if _index % 5000 == 0:
            print("have sampled %d pairs" % _index)
        q = pair[1]
        a = pair[2]
        distractor = pair[3]

        neg_alist_index = [i for i in range(len(alist))]
        sampled_index = np.random.choice(neg_alist_index, size=[FLAGS.pools_size], replace=False)
        pools = np.array(alist)[sampled_index]  # it's possible that true positive samples are selected
        # TODO: remove true positives

        # [q, a, distractor, negative sample]
        canditates = data_helpers.loadCandidateSamples(q, a, distractor, pools, vocab,
                                                       FLAGS.max_sequence_length_q, FLAGS.max_sequence_length_a)
        predicteds = []
        for batch in data_helpers.batch_iter(canditates, batch_size=FLAGS.batch_size):
            feed_dict = {
                model.input_x_1: np.array(batch[:, 0].tolist()),
                model.input_x_2: np.array(batch[:, 1].tolist()),
                model.input_x_3: np.array(batch[:, 2].tolist()),
                model.input_x_4: np.array(batch[:, 3].tolist())
            }
            predicted = sess.run(model.gan_score, feed_dict)
            predicteds.extend(predicted)

        predicteds = np.array(predicteds) * FLAGS.sampled_temperature
        predicteds -= np.max(predicteds)
        exp_rating = np.exp(predicteds)
        prob = exp_rating / np.sum(exp_rating)
        prob = np.nan_to_num(prob) + 1e-7
        prob = prob / np.sum(prob)
        neg_samples = np.random.choice(pools, size=negative_size, p=prob, replace=False)
        for neg in neg_samples:
            samples.append((encode_sent(vocab, q, FLAGS.max_sequence_length_q),
                            encode_sent(vocab, a, FLAGS.max_sequence_length_a),
                            encode_sent(vocab, distractor, FLAGS.max_sequence_length_a),
                            encode_sent(vocab, neg, FLAGS.max_sequence_length_a)))
    return samples

def sample_data(word_vocab, char_vocab, negative_size=FLAGS.gan_k):
    '''used for generate negative samples for the Discriminator'''
    samples = []

    word_input_ids_q = []
    word_input_ids_a = []
    word_input_ids_dis = []
    char_input_ids_q = []
    char_input_ids_a = []
    char_input_ids_dis = []
    labels = []
    tokenizer = word_tokenize

    for _index, pair in enumerate(raw):
        # if _index==100:
        #     break
        if _index % 5000 == 0:
            print("have sampled %d pairs" % _index)
        q = pair[1]
        a = pair[2]
        distractor = pair[3]

        neg_alist_index = [i for i in range(len(alist))]
        # sampled_index = np.random.choice(neg_alist_index, size=[FLAGS.pools_size], replace=False)
        sampled_index = np.random.choice(neg_alist_index, size=[negative_size], replace=False)
        pools = np.array(alist)[sampled_index]  # it's possible that true positive samples are selected
        pools = [' '.join(pool.split('_')) for pool in pools]
        pools = [re.sub('<a>','',pool).strip() for pool in pools]

        # TODO: remove true positives
        # word_samples_q = []
        # word_samples_a = []
        # word_samples_dis = []
        # char_samples_q = []
        # char_samples_a = []
        # char_samples_dis = []

        word_ids_q, char_ids_q = encode_sent_lstm(tokenizer, q, FLAGS.max_sent_len, vocab, FLAGS.max_word_len, char_vocab, FLAGS.conv_filter_size)
        word_ids_a, char_ids_a = encode_sent_lstm(tokenizer, a, FLAGS.max_sent_len, vocab, FLAGS.max_word_len, char_vocab, FLAGS.conv_filter_size)
        word_ids_dis, char_ids_dis = encode_sent_lstm(tokenizer, distractor, FLAGS.max_sent_len, vocab, FLAGS.max_word_len, char_vocab, FLAGS.conv_filter_size)        
        for neg in pools:
            word_input_ids_q.append(word_ids_q)
            char_input_ids_q.append(char_ids_q)
            word_input_ids_a.append(word_ids_a)
            char_input_ids_a.append(char_ids_a)
            word_input_ids_dis.append(word_ids_dis)
            char_input_ids_dis.append(char_ids_dis)
            labels.append(1)

            word_ids, char_ids = encode_sent_lstm(tokenizer, neg, FLAGS.max_sent_len, vocab, FLAGS.max_word_len, char_vocab, FLAGS.conv_filter_size)
            word_input_ids_q.append(word_ids_q)
            char_input_ids_q.append(char_ids_q)
            word_input_ids_a.append(word_ids_a)
            char_input_ids_a.append(char_ids_a)            
            word_input_ids_dis.append(word_ids)
            char_input_ids_dis.append(char_ids)
            labels.append(0)

    samples = (word_input_ids_q, word_input_ids_q, word_input_ids_dis, \
        char_input_ids_q, char_input_ids_q, char_input_ids_dis)

    return samples, labels


def get_metrics(target_list, batch_scores, topk=10):
    '''
    Args:
        target_list: [1, 1, 0, 0, ...]. Could be all zeros.
        batch_scores: [0.9, 0.7, 0.2, ...]. Predicted relevance.
    '''
    length = min(len(target_list), len(batch_scores))
    if length == 0:
        return [0, 0, 0, 0]
    target_list = target_list[:length]
    batch_scores = batch_scores[:length]
    target_list = np.array(target_list)
    predict_list = np.argsort(batch_scores)[::-1]
    predict_list = target_list[predict_list]
    # RR and AP for MRR and MAP
    RR = 0.0
    avg_prec = 0.0
    precisions = []
    num_correct = 0.0
    prec1 = 0.0
    num_correct_total = 0.0
    for i in range(len(predict_list)):
        if predict_list[i] == 1:
            num_correct_total += 1
    for i in range(min(topk, len(predict_list))):
        if i == 0:
            if predict_list[i] == 1:
                prec1 = 1.0
        if predict_list[i] == 1:
            if RR == 0:
                RR += 1.0 / (i + 1)
            num_correct += 1
            precisions.append(num_correct / (i + 1))
    if len(precisions) > 0:
        avg_prec = sum(precisions) / len(precisions)
    if num_correct_total == 0.0:
        recall = 0.0
    else:
        recall = num_correct / num_correct_total
    return [RR, avg_prec, prec1, recall]


@log_time_delta
def dev_step(model, devList, saveresults=False):
    # grouped by q
    testList_dict = {}
    tokenizer = word_tokenize

    for i in range(len(devList)):
        item = devList[i].split(' ')
        label, q = item[0], item[1]
        if q in testList_dict:
            testList_dict[q][0].append(int(label))
            testList_dict[q][1].append(devList[i])
        else:
            testList_dict[q] = [[int(label)], [devList[i]]]

    # save results
    saveresults_path = 'log/{}_results.txt'.format(FLAGS.prefix)
    if saveresults:
        with open(saveresults_path, 'w', encoding='utf8') as fout:
            fout.write('')

    # evaluation
    metrics_all = []
    for q, item in testList_dict.items():
        target_list = item[0]
        if np.sum(target_list) == 0:
            continue

        testList_sub = item[1]
        # batch_scores
        batch_scores = []
        for i in range(int(math.ceil(len(testList_sub) / FLAGS.batch_size))):
            if (i + 1) * FLAGS.batch_size > len(testList_sub):
                batch_size_real = len(testList_sub) - i * FLAGS.batch_size
            else:
                batch_size_real = FLAGS.batch_size
            batch_input_ids = data_helpers.load_val_batch_lstm(testList_sub, tokenizer,
                                                                                 i*FLAGS.batch_size,
                                                                                 batch_size_real,
                                                                                 vocab,
                                                                                 char_vocab,
                                                                                 FLAGS)
            b_input_data = [torch.tensor(b_input_ids, dtype=np.long) for b_input_ids in batch_input_ids]
            # b_labels = torch.tensor(b_labels, dtype=np.float)

            b_input_ids = [b_input_ids.to(device) for b_input_ids in b_input_data]
            # b_labels = b_labels.to(device)

            _, _, _, _, prob = model(b_input_ids)
            # prob = prob.reshape([-1,2])[:,0].detach().cpu()
            prob = prob.detach().cpu()
            # prob = 1 - prob
            batch_scores.extend(prob)

        # save results
        if saveresults:
            for j in range(len(batch_scores)):
                with open(saveresults_path, 'a', encoding='utf8') as fout:
                    fout.write('{} {} {}\n'.format(target_list[j],
                        testList_sub[j], batch_scores[j]))

        # MRR@10, MAP@10, Precision@1, Recall@10, etc.
        metrics = get_metrics(target_list, batch_scores, 10)
        metrics_all.append(metrics)
    metrics_all = np.array(metrics_all)
    # metrics_all = np.mean(metrics_all, axis=0)
    metrics_all = np.sum(metrics_all, axis=0) / len(testList_dict)
    return metrics_all.tolist()


@log_time_delta
def evaluation(model, log, num_epochs=0, split='dev',
        savemodel=True, justsave=False, saveresults=False):

    metrics_current = [0, 0, 0, 0]
    metrics_current = [str(x) for x in metrics_current]
    if justsave:
        filename = "model_lstm/{}_{}_{}.model".format(FLAGS.prefix, num_epochs, '_'.join(metrics_current))
        # saver.save(sess, filename)
        torch.save(model.state_dict(), filename)
        return

    assert split in ['dev', 'test', 'testall']
    if split == 'dev':
        metrics_current = dev_step(model, devList, saveresults)
    elif split == 'test':
        metrics_current = dev_step(model, testList, saveresults)
    elif split == 'testall':
        metrics_current = dev_step(model, testallList, saveresults)

    line = "test: %d epoch: metric %s" % (num_epochs, metrics_current)
    print(line)
    log.write(line + "\n")
    log.flush()
    metrics_current = [str(x) for x in metrics_current]

    if savemodel:
        filename = "model_lstm/{}_{}_{}.model".format(FLAGS.prefix, num_epochs, '_'.join(metrics_current))
        # saver.save(sess, filename)
        torch.save(model.state_dict(), filename)

class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds

class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, \
        drop_out_rate, conv_filter_size, word_embeddings, char_embed_dim, \
        max_word_len, char_vocab, char_feature_size):

        super(LstmModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate

        self.word_embeddings = nn.Embedding(word_embeddings.shape[0], word_embeddings.shape[1])
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embeddings.astype(np.double)))
        # self.word_embeddings.weight.data.requires_grad = True
        self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, self.drop_rate)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
          bidirectional=bool(self.is_bidirectional))

        # self.dropout = nn.Dropout(self.drop_rate)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1,\
                                     max_word_len + conv_filter_size - 1)


    def forward(self, words, chars):
        batch_size = words.shape[0]
        max_batch_len = words.shape[1]

        # words = words.view(words.shape[0]*words.shape[1],words.shape[2])
        # chars = chars.view(chars.shape[0]*chars.shape[1],chars.shape[2])

        src_word_embeds = self.word_embeddings(words)
        char_embeds = self.char_embeddings(chars)
        char_embeds = char_embeds.permute(0, 2, 1)

        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)

        words_input = torch.cat((src_word_embeds, char_feature), -1)
        outputs, hc = self.lstm(words_input)

        # h_drop = self.dropout(hc[0])
        h_n = hc[0].view(self.layers, 2, words.shape[0], self.hidden_dim)
        h_n = h_n[-1,:,:,:] # (num_dir,batch,hidden)
        h_n = h_n.permute((1,0,2)) # (batch,num_dir,hidden)
        h_n = h_n.contiguous().view(h_n.shape[0],h_n.shape[1]*h_n.shape[2]) # (batch,num_dir*hidden)

        return h_n, words_input

class BaseModel(nn.Module):
    def __init__(self, word_embeddings, char_vocab):
        super(BaseModel, self).__init__()
        self.encoder = LstmModel(FLAGS.input_dim, FLAGS.hidden_dim, FLAGS.layers, FLAGS.is_bidirectional, FLAGS.\
        drop_out_rate, FLAGS.conv_filter_size, word_embeddings, FLAGS.char_embed_dim, FLAGS.max_word_len, char_vocab, FLAGS.char_feature_size)
        # self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(FLAGS.hidden_dim*2*2 + 2*FLAGS.max_sent_len, 2)
        # self.linear = nn.Linear(FLAGS.hidden_dim*2, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)

    def forward(self, input_ids):

        words_q, words_a, words_dis, chars_q, chars_a, chars_dis = input_ids
        enc_q, emb_q = self.encoder.forward(words_q, chars_q)
        enc_a, emb_a = self.encoder.forward(words_a, chars_a)
        enc_d, emb_d = self.encoder.forward(words_dis, chars_dis)

        cost, C, P = self.sinkhorn(emb_a, emb_d)
        min_pair = torch.min(P, dim=-1)[1]
        indices = min_pair.unsqueeze(-1).repeat([1,1,emb_a.shape[2]])
        emb_d_coupled = torch.gather(emb_d, dim=1, index=indices)
        emb_diff = emb_a - emb_d_coupled
        emb_diff_norm_d = torch.norm(emb_diff, dim=2)

        cost, C, P = self.sinkhorn(emb_a, emb_q)
        min_pair = torch.min(P, dim=-1)[1]
        indices = min_pair.unsqueeze(-1).repeat([1,1,emb_a.shape[2]])
        emb_q_coupled = torch.gather(emb_q, dim=1, index=indices)
        emb_diff = emb_a - emb_q_coupled
        emb_diff_norm_q = torch.norm(emb_diff, dim=2)

        if BINARY_CLASSIFICATION:
            logit = torch.sum((enc_q+enc_a) * enc_d, dim=-1)
            prob = torch.nn.Sigmoid()(logit)

            return enc_q, enc_a, enc_d, logit, prob
        else:
            logit_1 = enc_q + enc_d
            logit_2 = enc_a + enc_d
            logit = torch.cat((logit_1,logit_2,emb_diff_norm_d,emb_diff_norm_q), dim=-1)
            # logit = enc_a + enc_d
            logit = self.linear(logit)
            prob = self.softmax(logit)

            return enc_q, enc_a, enc_d, logit, prob[:,1]

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        if torch.cuda.is_available():
            mu = torch.empty(batch_size, x_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
            nu = torch.empty(batch_size, y_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()
        else:
            mu = torch.empty(batch_size, x_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / x_points).squeeze()
            nu = torch.empty(batch_size, y_points, dtype=torch.float,
                             requires_grad=False).fill_(1.0 / y_points).squeeze()

        if torch.cuda.is_available():
            u = torch.zeros_like(mu).cuda()
            v = torch.zeros_like(nu).cuda()
        else:
            u = torch.zeros_like(mu)
            v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

def flat_accuracy(preds, labels):
    pred_flat = np.where(preds>0.5, 1., 0.)
    # pred_flat = np.argmax(preds, axis=1).flatten()
    # labels_flat = labels.flatten()
    return np.sum(pred_flat == labels) / len(labels)

def main():
    # embeddings
    param = None
    if len(FLAGS.pretrained_embeddings_path) > 0:
        print('loading pretrained embeddings...')
        param = embd
    else:
        print('using randomized embeddings...')
        param = np.random.uniform(-0.05, 0.05, (len(vocab), FLAGS.embedding_dim))

    # models
    model = BaseModel(embd, char_vocab)
    if CUDA:
        model.cuda()
    if BINARY_CLASSIFICATION:
        if torch.cuda.is_available():
          loss_func = nn.BCELoss().cuda()
        else:
          loss_func = nn.BCELoss()
    else:
        if CUDA:
          loss_func = nn.CrossEntropyLoss().cuda()
        else:
          loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    with  open(log_precision, "w") as log, open(log_loss, "w") as loss_log:
        # initialze or restore


        for i in range(FLAGS.num_epochs):
            # discriminator
            # model.train()
            samples, labels = sample_data(vocab, char_vocab, FLAGS.gan_k)
            for _index, batch_samples in enumerate(data_helpers.batch_iter_lstm(samples,
                                                                   labels,
                                                                   FLAGS.batch_size,
                                                                   num_epochs=FLAGS.d_epochs_num,
                                                                   shuffle=True)):

                model.train()
                optimizer.zero_grad()

                b_input_data, b_labels = batch_samples[0], batch_samples[1]


                b_input_data = [torch.tensor(b_input_ids, dtype=np.long) for b_input_ids in b_input_data]
                b_labels = torch.tensor(b_labels, dtype=np.float)

                b_input_ids = [b_input_ids.to(device) for b_input_ids in b_input_data]
                b_labels = b_labels.to(device)

                _, _, _, logit, prob = model(b_input_ids)
                # if CUDA:
                #     prob = torch.tensor(prob, dtype=np.float).cuda()
                # else:
                #     prob = torch.tensor(prob, dtype=np.float)

                if BINARY_CLASSIFICATION:
                    loss = loss_func(prob.double(), b_labels)
                else:
                    loss = loss_func(logit, b_labels.long())

                loss.backward()
                optimizer.step()

                accuracy = flat_accuracy(prob.cpu().detach().numpy(), b_labels.cpu().detach().numpy())
                # if BINARY_CLASSIFICATION:
                #     accuracy = flat_accuracy(prob.cpu().detach().numpy(), b_labels.cpu().detach().numpy())
                # else:
                #     accuracy = flat_accuracy(prob[:,1].cpu().detach().numpy(), b_labels.cpu().detach().numpy())

                line = ("%s: step %d, loss %f with acc %f" % (
                    datetime.datetime.now().isoformat(),
                    _index, loss.item(), accuracy)
                )
                if _index % 100 == 0:
                    print(line)
                    print(prob)
                    print(b_labels)
                    print()
                loss_log.write(line+"\n")
                loss_log.flush()
            evaluation(model, log, i,
            'dev', True, False)


        # final evaluation
        evaluation(model, log, -1, 'test', False,
                False, True)

if __name__ == '__main__':
    main()
