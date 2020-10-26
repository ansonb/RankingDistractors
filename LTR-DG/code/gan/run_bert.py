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

from data_helpers import encode_sent, encode_sent_bert
import data_helpers as data_helpers

from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertModel
from torch import nn, optim
import torch
from keras.preprocessing.sequence import pad_sequences

import re

# import dataHelper
# Data
tf.flags.DEFINE_string("dataset", "semevalQA", "dataset path")
tf.flags.DEFINE_string("prefix", "semevalQA", "prefix")
# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length_q", 100, "Max sequence length fo sentence (default: 100)")
tf.flags.DEFINE_integer("max_sequence_length_a", 28, "Max sequence length fo sentence (default: 100)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("hidden_size", 300, "Dimensionality of character embedding (default: 128)")
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

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
assert(FLAGS.batch_size == FLAGS.pools_size)

MAX_SENT_LEN = 128

print(("\nParameters:"))
for attr, value in sorted(FLAGS.__flags.items()):
        print(("{}={}".format(attr.upper(), value)))
print((""))

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
# device = "cpu"

print(("Loading data..."))
vocab, embd = data_helpers.build_vocab(FLAGS.dataset, FLAGS.pretrained_embeddings_path)
if len(FLAGS.pretrained_embeddings_path) > 0:
    assert(embd.shape[1] == FLAGS.embedding_dim)
    with open('{}/embd.pkl'.format(FLAGS.dataset), 'wb') as fout:
        pickle.dump(embd, fout)
with open('{}/vocab.pkl'.format(FLAGS.dataset), 'wb') as fout:
    pickle.dump(vocab, fout)
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

def sample_data(negative_size=FLAGS.gan_k):
    '''used for generate negative samples for the Discriminator'''
    samples = []

    input_ids_qa = []
    input_ids_dis = []
    input_ids_neg = []
    attention_masks_qa = []
    attention_masks_dis = []
    attention_masks_neg = []

    for _index, pair in enumerate(raw):
        if _index==100:
            break
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

        # # [q, a, distractor, negative sample]
        # canditates = data_helpers.loadCandidateSamples(q, a, distractor, pools, vocab,
        #                                                FLAGS.max_sequence_length_q, FLAGS.max_sequence_length_a)
        # predicteds = []
        # for batch in data_helpers.batch_iter(canditates, batch_size=FLAGS.batch_size):
        #     feed_dict = {
        #         model.input_x_1: np.array(batch[:, 0].tolist()),
        #         model.input_x_2: np.array(batch[:, 1].tolist()),
        #         model.input_x_3: np.array(batch[:, 2].tolist()),
        #         model.input_x_4: np.array(batch[:, 3].tolist())
        #     }
        #     predicted = sess.run(model.gan_score, feed_dict)
        #     predicteds.extend(predicted)

        # predicteds = np.array(predicteds) * FLAGS.sampled_temperature
        # predicteds -= np.max(predicteds)
        # exp_rating = np.exp(predicteds)
        # prob = exp_rating / np.sum(exp_rating)
        # prob = np.nan_to_num(prob) + 1e-7
        # prob = prob / np.sum(prob)
        # neg_samples = np.random.choice(pools, size=negative_size, p=prob, replace=False)
        samples_qa = []
        samples_dis = []
        samples_neg = []
        # for neg in neg_samples:
        #     samples.append((encode_sent_bert(tokenizer, q, FLAGS.max_sequence_length_q, answer=a),
        #                     encode_sent_bert(tokenizer, a, FLAGS.max_sequence_length_a),
        #                     encode_sent_bert(tokenizer, distractor, FLAGS.max_sequence_length_a),
        #                     encode_sent_bert(tokenizer, neg, FLAGS.max_sequence_length_a)))
        _q = '[CLS] {} [SEP] {} [SEP]'.format(q,a)
        _distractor = '[CLS] {} [SEP]'.format(distractor)
        for neg in pools:
            _neg = '[CLS] {} [SEP]'.format(neg)
            samples_qa.append(encode_sent_bert(tokenizer, _q))
            samples_dis.append(encode_sent_bert(tokenizer, _distractor))
            samples_neg.append(encode_sent_bert(tokenizer, _neg))

        input_ids_qa_batch = pad_sequences(samples_qa, maxlen=MAX_SENT_LEN, dtype="long", truncating="post", padding="post")
        input_ids_dis_batch = pad_sequences(samples_dis, maxlen=MAX_SENT_LEN, dtype="long", truncating="post", padding="post")
        input_ids_neg_batch = pad_sequences(samples_neg, maxlen=MAX_SENT_LEN, dtype="long", truncating="post", padding="post")

        # Create attention masks
        attention_masks_qa_batch = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids_qa_batch:
          seq_mask = [float(i>0) for i in seq]
          attention_masks_qa_batch.append(seq_mask)

        # Create attention masks
        attention_masks_dis_batch = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids_dis_batch:
          seq_mask = [float(i>0) for i in seq]
          attention_masks_dis_batch.append(seq_mask)

        # Create attention masks
        attention_masks_neg_batch = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids_neg_batch:
          seq_mask = [float(i>0) for i in seq]
          attention_masks_neg_batch.append(seq_mask)

        input_ids_qa.extend(input_ids_qa_batch)
        input_ids_dis.extend(input_ids_dis_batch)
        input_ids_neg.extend(input_ids_neg_batch)
        attention_masks_qa.extend(attention_masks_qa_batch)
        attention_masks_dis.extend(attention_masks_dis_batch)
        attention_masks_neg.extend(attention_masks_neg_batch)

    samples = (input_ids_qa, input_ids_dis, input_ids_neg)
    attention_masks = (attention_masks_qa, attention_masks_dis, attention_masks_neg)

    import pdb; pdb.set_trace()
    return samples, attention_masks


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
            b_input_ids, b_input_mask = data_helpers.load_val_batch_bert(testList_sub, tokenizer,
                                                                                 i*FLAGS.batch_size,
                                                                                 batch_size_real,
                                                                                 MAX_SENT_LEN=MAX_SENT_LEN)
            b_input_ids = torch.tensor(b_input_ids, dtype=np.long)
            b_input_mask = torch.tensor(b_input_mask, dtype=np.long)
            # b_labels = torch.tensor(b_labels)

            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            # b_labels = b_labels.to(device)

            _, _, prob = model(b_input_ids, b_input_mask)
            prob = prob.reshape([-1,2])[:,0].detach().cpu()
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
        filename = "model_bert/{}_{}_{}.model".format(FLAGS.prefix, num_epochs, '_'.join(metrics_current))
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
        filename = "model_bert/{}_{}_{}.model".format(FLAGS.prefix, num_epochs, '_'.join(metrics_current))
        # saver.save(sess, filename)
        torch.save(model.state_dict(), filename)

class BertBaseModel(nn.Module):
    def __init__(self):
        super(BertBaseModel, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-cased")
        # self.dropout = nn.Dropout(drop_rate)
        # self.linear = nn.Linear(768,NUM_LABELS)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder.forward(input_ids=input_ids,attention_mask=attention_mask)
        enc_hs = outputs[0][-1][:,0,:]  # The last hidden-state is the first element of the output tuple
        # enc_hs = outputs[1]
        enc_hs = enc_hs.reshape(enc_hs.shape[0]//3,3,768) # append the embeddings due to the sentence, subject entity context and object entity context
        # rel = self.softmax(self.linear(enc_hs))

        # return rel, enc_hs
        enc_hs_qa = enc_hs[:,0,:]
        enc_hs_dis = enc_hs[:,1,:]
        enc_hs_neg = enc_hs[:,2,:]

        pos_logit = torch.sum(enc_hs_qa * enc_hs_dis, dim=-1).unsqueeze(-1)
        pos_proba = torch.nn.Sigmoid()(pos_logit)

        neg_logit = torch.sum(enc_hs_qa * enc_hs_neg, dim=-1).unsqueeze(-1)
        neg_proba = torch.nn.Sigmoid()(neg_logit)

        # print(pos_logit)
        # print(neg_logit)

        logit = torch.cat([pos_logit,neg_logit], dim=-1).reshape((-1))
        prob = torch.cat([pos_proba,neg_proba], dim=-1).reshape((-1))

        return enc_hs, logit, prob

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
    model = BertBaseModel()
    if CUDA:
        model.cuda()
    if torch.cuda.is_available():
      loss_func = nn.BCELoss().cuda()
    else:
      loss_func = nn.BCELoss()
    param_optimizer_bert = list(model.encoder.named_parameters())
    param_optimizer_bert_names = [n for n, p in param_optimizer_bert]
    param_optimizer_all = list(model.named_parameters())
    param_optimizer_all_names_excluding_bert = [n for n, p in param_optimizer_all if n[8:] not in param_optimizer_bert_names]
    print('== param_optimizer_all_names_excluding_bert')
    print(param_optimizer_all_names_excluding_bert)
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer_bert if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01,
         'lr': 2e-5},
        {'params': [p for n, p in param_optimizer_bert if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0,
         'lr': 2e-5},
         {'params': [p for n, p in param_optimizer_all if n in param_optimizer_all_names_excluding_bert],
         'lr': 1e-3,
         'weight_decay_rate': 0.01}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = optim.Adam(optimizer_grouped_parameters)
    with  open(log_precision, "w") as log, open(log_loss, "w") as loss_log:
        # initialze or restore


        for i in range(FLAGS.num_epochs):
            # discriminator
            model.train()
            samples, attention_masks = sample_data(FLAGS.gan_k)
            for _index, batch_samples in enumerate(data_helpers.batch_iter_bert(samples,
                                                                   attention_masks,
                                                                   FLAGS.batch_size,
                                                                   num_epochs=FLAGS.d_epochs_num,
                                                                   shuffle=True)):

                optimizer.zero_grad()

                b_input_ids, b_input_mask, b_labels = batch_samples[0], batch_samples[1], batch_samples[2]

                b_input_ids = torch.tensor(b_input_ids, dtype=np.long)
                b_input_mask = torch.tensor(b_input_mask, dtype=np.long)
                b_labels = torch.tensor(b_labels, dtype=np.float)

                b_input_ids = b_input_ids.to(device)
                b_input_mask = b_input_mask.to(device)
                b_labels = b_labels.to(device)

                _, logit, prob = model(b_input_ids, b_input_mask)
                # if CUDA:
                #     prob = torch.tensor(prob, dtype=np.float).cuda()
                # else:
                #     prob = torch.tensor(prob, dtype=np.float)

                loss = loss_func(prob.double(), b_labels)

                loss.backward()
                optimizer.step()

                accuracy = flat_accuracy(prob.cpu().detach().numpy(), b_labels.cpu().detach().numpy())

                line = ("%s: step %d, loss %f with acc %f" % (
                    datetime.datetime.now().isoformat(),
                    _index, loss.item(), accuracy)
                )
                if _index % 12 == 0:
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
