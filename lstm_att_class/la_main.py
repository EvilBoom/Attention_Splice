# _*_ coding: utf-8 _*_
# @Time : 2021/6/8 11:02
# @Author : 张宝宇
# @Version：V 0.0
# @File : la_main.py
# @desc :
import pandas as pd
import torch
from torchtext import data
import spacy
from spacy.tokenizer import Tokenizer
import time

import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm

SEED = 1234
N_EPOCHS = 10


def spacy_tokenize(x):
    return [tok.text for tok in tokenizer(x)]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mines = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mines * 60))
    return elapsed_mines, elapsed_secs


def count_parameters(c_model):
    return sum(p.numel() for p in c_model.parameters() if p.requires_grad)


def binary_accuracy(pred_s, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_pred_s = torch.round(torch.sigmoid(pred_s))
    bin_correct = (rounded_pred_s == y).float()  # convert into float for division
    bin_acc = bin_correct.sum() / len(bin_correct)
    return bin_acc


def get_ROC_curve(apredicted_labels, probabilities):
    dat = np.hstack((np.array(apredicted_labels).reshape(-1, 1), np.array(probabilities).reshape(-1, 1)))
    dataset = pd.DataFrame(dat, columns=["y", "proba"])
    # lists to store TPR and FRP values
    TPR_values = []
    FPR_values = []
    nrows = len(dataset)
    # Sort dataframe in descending order by probability values
    dataset.sort_values("proba", ascending=False, inplace=True)
    # A list that stores the actual class labels
    act_list = dataset["y"].tolist()
    # A list that stores the probability scores
    prob_list = dataset["proba"].tolist()
    # Calculate the number of positive values
    P = dataset["y"].tolist().count(1)
    # Calculate the number of negative values
    N = dataset["y"].tolist().count(0)
    for i in range(0, nrows):
        # Select the threshold
        thresh = prob_list[i]
        # Initialize TP and FP to zero
        TP = FP = 0
        # Calculate The number of true positives and number of false positive for each threshold
        for j in range(0, nrows):
            class_label = 1 if prob_list[j] >= thresh else 0
            actual_label = act_list[j]
            if class_label == 1:
                if actual_label == 1:
                    TP += 1
                else:
                    FP += 1
        # Append them to the array
        TPR_values.append(TP / P)
        FPR_values.append(FP / N)
    # Plot the ROC Curve
    print("The ROC Curve is:")
    plt.plot(FPR_values, TPR_values, label="ROC Curve")
    plt.grid()
    plt.legend()
    plt.show()


# def get_PRC_Curve(y_test_probs, actual):
#     probability_thresholds = np.linspace(0, 1, num=100)
#     precision_scores = []
#     recall_scores = []
#     for p in probability_thresholds:
#         y_test_preds = []
#         for prob in y_test_probs:
#             if prob > p:
#                 y_test_preds.append(1)
#             else:
#                 y_test_preds.append(0)
#         precision, recall = calc_precision_recall(actual, y_test_preds)
#         precision_scores.append(precision)
#         recall_scores.append(recall)
#     plt.plot(precision_scores, label='PRECISION', color='red')
#     plt.plot(recall_scores, label='RECALL', color='green')
#     plt.plot(recall_scores, precision_scores, color='green', label='PR Curve')
#     plt.legend()
#     plt.grid()
#     plt.plot()


def confusion_matrix(actual, predict):
    tp, tn, fp, fn = 0, 0, 0, 0
    for a, b in zip(actual, predict):
        if a == b == 1:
            tp += 1
        elif a == b == 0:
            tn += 1
        elif a == 1 and b == 0:
            fn += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def evaluate(test_model, iterator, test_criterion):
    test_epoch_loss = 0
    test_epoch_acc = 0
    test_model.eval()
    with torch.no_grad():
        for test_batch in iterator:
            # text, text_lengths = batch.text
            test_batch_size = 64
            test_predictions = test_model(test_batch.text.to(device), test_batch_size).squeeze(1)
            test1_loss = test_criterion(test_predictions, test_batch.label.to(device))
            test1_acc = binary_accuracy(test_predictions, test_batch.label.to(device))
            test_epoch_loss += test1_loss.item()
            test_epoch_acc += test1_acc.item()
    return test_epoch_loss / len(iterator), test_epoch_acc / len(iterator)


def evaluate1(eval_model, iterator, eval_criterion):
    epoch_loss = 0
    epoch_acc = 0
    eval_model.eval()
    with torch.no_grad():
        for batch in iterator:
            # text, text_lengths = batch.text
            eval_batch_size = 64
            predictions = eval_model(batch.text.to(device), eval_batch_size).squeeze(1)
            loss = eval_criterion(predictions, batch.label.to(device))
            acc = binary_accuracy(predictions, batch.label.to(device))
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_mod(train_model, iterator, train_optimizer, train_criterion):
    epoch_loss = 0
    epoch_acc = 0
    train_model.train()
    for batch in tqdm(iterator):
        train_optimizer.zero_grad()
        # text, text_lengths = batch.text
        train_batch_size = 64
        predictions = train_model(batch.text.to(device), train_batch_size).squeeze(1)
        loss = train_criterion(predictions, batch.label.to(device))
        acc = binary_accuracy(predictions, batch.label.to(device))
        loss.backward()
        train_optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train1(train1_model, iterator, train1_optimizer, train1_criterion):
    epoch_loss = 0
    epoch_acc = 0
    train1_model.train()
    for batch in tqdm(iterator):
        train1_optimizer.zero_grad()
        # text, text_lengths = batch.text
        train1_batch_size = 64
        predictions = train1_model(batch.text.to(device), train1_batch_size).squeeze(1)
        loss = train1_criterion(predictions, batch.label.to(device))
        acc = binary_accuracy(predictions, batch.label.to(device))
        loss.backward()
        train1_optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


class BLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, batch_size, dropout, pad_idx):
        # vocab_size 25002 |  EMBEDDING_DIM 100 | HIDDEN_DIM = 256 | OUTPUT_DIM = 1 | N_LAYERS = 2
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, bl_batch_size):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        h_0 = Variable(torch.zeros(2 * self.n_layers, bl_batch_size, self.hidden_dim).to(device))
        c_0 = Variable(torch.zeros(2 * self.n_layers, bl_batch_size, self.hidden_dim).to(device))
        output, (hidden, final_cell_state) = self.rnn(embedded, (h_0, c_0))
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)


class ABLSTM1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, batch_size, dropout, pad_idx):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, al_batch_size):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))
        # embedded = embedded.permute(1,0,2)

        # embedded = [sent len, batch size, emb dim]
        h_0 = Variable(torch.zeros(2 * self.n_layers, al_batch_size, self.hidden_dim).to(device))
        c_0 = Variable(torch.zeros(2 * self.n_layers, al_batch_size, self.hidden_dim).to(device))
        output, (hidden, final_cell_state) = self.rnn(embedded, (h_0, c_0))

        output = output.permute(1, 0, 2)
        # print(output.shape)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        hidden = hidden.squeeze(0)
        # print(hidden.shape)
        # print(lstm_output.shape)

        # attention part
        attn_weights = torch.bmm(output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # print(hidden.shape)
        # attn_output = self.attention_net(output,hidden)
        # hidden = [batch size, hid dim * num directions]

        return self.fc(new_hidden_state)


if __name__ == '__main__':
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    nlp = spacy.load("en_core_web_sm")
    tokenizer = Tokenizer(nlp.vocab)
    TEXT = data.Field(tokenize=spacy_tokenize)
    LABEL = data.LabelField(dtype=torch.float)
    # train  40000  test 6400  valid 6400
    # give the format of text and label
    tv_data_fields = [("text", TEXT), ("label", LABEL)]
    train, valid, test = data.TabularDataset.splits(path="",
                                                    train="train.csv", validation="val.csv",
                                                    test="test.csv", format="csv",
                                                    skip_header=True, fields=tv_data_fields)
    # initial weight define by torchtext is zero but using this we can start it with random values
    MAX_VOCAB_SIZE = 25_000
    TEXT.build_vocab(train,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.300d",
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train)
    # convert data into the iterable form
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train, valid, test),
                                                                               batch_size=batch_size,
                                                                               sort_key=lambda x: len(x.label),
                                                                               sort_within_batch=True,
                                                                               device=device)
    INPUT_DIM = len(TEXT.vocab)
    print(INPUT_DIM)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    batch_size = 64

    model = BLSTM(INPUT_DIM,
                  EMBEDDING_DIM,
                  HIDDEN_DIM,
                  OUTPUT_DIM,
                  N_LAYERS,
                  BIDIRECTIONAL,
                  batch_size,
                  DROPOUT,
                  PAD_IDX)
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    batch_size = 64

    model1 = ABLSTM1(INPUT_DIM,
                     EMBEDDING_DIM,
                     HIDDEN_DIM,
                     OUTPUT_DIM,
                     N_LAYERS,
                     BIDIRECTIONAL,
                     batch_size,
                     DROPOUT,
                     PAD_IDX)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    pretrained_embeddings = TEXT.vocab.vectors
    print(pretrained_embeddings.shape)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    print(model.embedding.weight.data)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.0001)
    criterion1 = nn.BCEWithLogitsLoss()
    model1 = model1.to(device)
    criterion1 = criterion1.to(device)
    # train_losses = []
    # valid_losses = []
    # best_valid_loss = float('inf')
    # for epoch in range(N_EPOCHS):
    #     start_time = time.time()
    #     train_loss, train_acc = train_mod(model, train_iterator, optimizer, criterion)
    #     valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    #     end_time = time.time()
    #     epoch_mines, epoch_secs = epoch_time(start_time, end_time)
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(model.state_dict(), 'BLSTM-model.pt')
    #     train_losses.append(train_loss)
    #     valid_losses.append(valid_loss)
    #     print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mines}m {epoch_secs}s')
    #     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    #     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    # plt.plot(train_losses)
    # plt.plot(valid_losses)
    train_loss_a = []
    valid_loss_a = []
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_mod(model1, train_iterator, optimizer1, criterion1)
        valid_loss, valid_acc = evaluate(model1, valid_iterator, criterion1)
        end_time = time.time()
        eval_epoch_mines, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'ABLSTM-model.pt')
        train_loss_a.append(train_loss)
        valid_loss_a.append(valid_loss)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {eval_epoch_mines}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    plt.plot(train_loss_a)
    plt.plot(valid_loss_a)
    model.load_state_dict(torch.load('BLSTM-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    model.load_state_dict(torch.load('ABLSTM-model.pt'))
    test_loss, test_acc = evaluate(model1, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    batch_size = 64
    predicted_labels = []
    prediction_probabilities = []
    actual_labels = []
    predicted = []
    num_correct = 0
    total_images = 0
    confusion_mat = np.zeros([200, 200], int)  # initializing the confusion matrix
    with torch.no_grad():
        for batch in test_iterator:
            # inputs = batch.text.to(device)
            labels = batch.label.to(device)
            outputs = model(batch.text.to(device), batch_size).squeeze(1)
            prediction_probabilities += list(np.array(outputs.detach().cpu()))
            loss = criterion(outputs, batch.label.to(device))
            # outputs  = model(inputs,batch_size)
            # loss = criterion(outputs,labels)
            pred = torch.round(outputs.squeeze())
            predicted_labels += list(np.array(pred.detach().cpu(), dtype=int))
            actual_labels += list(np.array(labels.detach().cpu(), dtype=int))
            # compare predictions to true label
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)
            # _,predicted = torch.max(outputs,1)
            total_images += labels.size(0)
            # total_correct += (pred == labels).sum().item()
            # for i, l in enumerate(labels):
            # confusion_mat[l.item(), pred[i].item()] += 1          # Row = actual class , COl = predicted class
    model_accuracy = num_correct / total_images * 100
    print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    prediction_probabilities = []
    actual_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in test_iterator:
            # text, text_lengths = batch.text
            batch_size = 64
            predictions = model(batch.text.to(device), batch_size).squeeze(1)
            prediction_probabilities += list(np.array(predictions.detach().cpu()))
            loss = criterion(predictions, batch.label.to(device))
            pred = torch.round(torch.sigmoid(predictions))
            # pred = torch.round(outputs.squeeze())
            predicted_labels += list(np.array(pred.detach().cpu(), dtype=int))
            actual_labels += list(np.array(batch.label.detach().cpu(), dtype=int))
            acc = binary_accuracy(predictions, batch.label.to(device))
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    # return epoch_loss / len(iterator), epoch_acc / len(iterator)
    TP, TN, FP, FN = confusion_matrix(actual_labels, predicted_labels)
    print([[TP, FP], [FN, TN]])
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    prediction_probabilities1 = []
    actual_labels1 = []
    predicted_labels1 = []
    with torch.no_grad():
        for batch in test_iterator:
            # text, text_lengths = batch.text
            batch_size = 64
            predictions = model1(batch.text.to(device), batch_size).squeeze(1)
            prediction_probabilities1 += list(np.array(predictions.detach().cpu()))
            loss = criterion(predictions, batch.label.to(device))
            pred = torch.round(torch.sigmoid(predictions))
            # pred = torch.round(outputs.squeeze())
            predicted_labels1 += list(np.array(pred.detach().cpu(), dtype=int))
            actual_labels1 += list(np.array(batch.label.detach().cpu(), dtype=int))
            acc = binary_accuracy(predictions, batch.label.to(device))
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    # return epoch_loss / len(iterator), epoch_acc / len(iterator)
    TP, TN, FP, FN = confusion_matrix(actual_labels1, predicted_labels1)
    print([[TP, FP], [FN, TN]])
    get_ROC_curve(predicted_labels, prediction_probabilities)
    get_ROC_curve(predicted_labels1, prediction_probabilities1)