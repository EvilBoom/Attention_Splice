import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import os
def PlotFigure(result,save,use_save=False):
    train_loss = result['train loss']
    valid_loss = result['valid loss']
    train_acc = result['train acc']
    valid_acc = result['valid acc']

    fig = plt.figure(1)

    font = {'family' : 'serif', 'color'  : 'black', 'weight' : 'bold', 'size'   : 16,}


    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(train_loss, 'r', label='Training Loss')
    ln2 = ax1.plot(valid_loss, 'k', label='Validation Loss')
    ax2 = ax1.twinx()
    ln3 = ax2.plot(train_acc, 'r--', label='Training Accuracy')
    ln4 = ax2.plot(valid_acc, 'k--', label='Validation Accuracy')

    lns = ln1+ ln2+ ln3+ ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=7)


    ax1.set_ylabel('Loss', fontdict=font)
    ax1.set_title("All Donor Splice Site Classification", fontdict=font)
    ax1.set_xlabel('Epoch', fontdict=font)

    ax2.set_ylabel('Accuracy', fontdict=font)

    plt.show()
    if use_save:
        figname = 'ALL_DON_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.png'
        fig.savefig(os.path.join(save,figname))
        print('Figure %s is saved.' % figname)
if __name__=='__main__':
    PlotFigure(result, save,use_save=True)
