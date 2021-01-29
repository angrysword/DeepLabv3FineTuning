from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data

import datahandler
from model import createDeepLabv3
from trainer import train_model
'''
@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=25,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
'''
def load_chk_data(chkpath,model,optimizer):
    checkpoint = torch.load(chkpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("load chk model:{},loss:{}".format(chkpath,loss))



def main(data_directory, exp_directory, epochs, batch_size,chkpath=None,loadchk=False):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    #criterion = torch.nn.CrossEntropyLoss(reduction='mean') #
    #loss function change 1/3
    criterion = torch.nn.BCELoss()
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if loadchk:
        load_chk_data(chkpath,model,optimizer)

    model.train()

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')


if __name__ == "__main__":
    #data_directory ='./CrackForest'
    #exp_directory='./CFExp'
    data_directory ='./FloorData'
    exp_directory='./FloorExp'
    #exp_directory='../drive/MyDrive/aexp'
    epochs=15
    batch_size=2
    
    chkpath='.\FloorExp\Jan19.pt'
    main(data_directory,exp_directory,epochs,batch_size,chkpath,False)
