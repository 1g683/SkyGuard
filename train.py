import gc
import os
import argparse
import time
import datetime
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import net
import config
from Dataset import SatUAVDataset
from utils import data_transforms
from transformers import ViTImageProcessor

model_names = sorted(name for name in net.__dict__ if name.startswith("Siamese") and callable(net.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=25,  help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16,  help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--step', type=int, default=10, help='learning rate step size')
parser.add_argument('--margin', type=float, default=4, help='margin of Cosine Similarity')
parser.add_argument('--data', default='raw', choices=['raw', 'aug', ], help='only raw data or with augmented data')
parser.add_argument('--model', default='SiameseEfficientNet', choices=model_names, help='model architecture: ' + ' | '.join(model_names))
opt = parser.parse_args()
print(opt)

def train_model(model, dataloaders, device, criterion, optimizer, scheduler, time_str, num_epochs=25,):
    print( model.__class__.__name__, 'starts to train.')
    train_losses = []
    val_losses = []
    train_start_time = time.time()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            Siamese_acc = {'TP':0, 'TN':0, 'FP':0, 'FN':0}

            # Iterate over data.
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                A = sample_batched['A'].to(device)
                B = sample_batched['B'].to(device)
                labels = sample_batched['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(A, B)
                    loss = criterion(outputs, labels)
                    if model.__class__.__name__ in ['SiameseEfficientNet', 'SiameseResNet', 'SiameseSqueezeNet', 'SiameseViTNet', 'SiameseMobileNet']:
                        dist = F.pairwise_distance(outputs[0], outputs[1])
                        preds = (dist.cpu().data.numpy()[:, np.newaxis] > 2)*1
                        Siamese_acc['TP'] += np.sum(np.logical_and(labels.cpu().data.numpy()==preds, preds==1))
                        Siamese_acc['TN'] += np.sum(np.logical_and(labels.cpu().data.numpy()==preds, preds==0))
                        Siamese_acc['FP'] += np.sum(np.logical_and(labels.cpu().data.numpy()!=preds, preds==1))
                        Siamese_acc['FN'] += np.sum(np.logical_and(labels.cpu().data.numpy()!=preds, preds==0))
                    else:
                        preds = (outputs.cpu().data.numpy() > 0.5) * 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * sample_batched['A'].size(0)
                running_corrects += torch.sum(torch.from_numpy(preds) == labels.cpu().long())

                # For DEBUG
                print("batch:%d/%d, loss:%.4f" % (i_batch, len(dataloaders[phase]), loss.item() * sample_batched['A'].size(0)), end='  |  ', flush=True)
                def rs(s):
                    return " ".join(str(s).replace('\n', ' ').split())
                if (1+epoch) % 5 == 2:
                    if model.__class__.__name__ in ['SiameseEfficientNet', 'SiameseResNet', 'SiameseSqueezeNet', 'SiameseViTNet','SiameseMobileNet']:
                        data_str=('%s, %s, %s, %s' % ( rs(dist.cpu().data), rs(preds), rs(labels.cpu().data), torch.sum(torch.from_numpy(preds) == labels.cpu().long())))
                    else:
                        data_str = ('%s, %s, %s' % (rs(outputs.cpu().data), rs(preds), rs(labels.cpu().data)))
                    print(data_str)

                # save memory to avoid memory usage exceeds limitation
                del A, B, outputs, loss, labels
                gc.collect()

            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if model.__class__.__name__ in ['SiameseEfficientNet', 'SiameseResNet', 'SiameseSqueezeNet', 'SiameseViTNet','SiameseMobileNet']:
                print("\n%s, TPR(Paired acc):%.2f, TNR(Unpaired acc):%.2f" %
                    (phase, Siamese_acc['TP']/(Siamese_acc['TP']+Siamese_acc['FN']),
                    Siamese_acc['TN']/(Siamese_acc['TN']+Siamese_acc['FP']),), end=' | ')
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format( phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                train_losses.append(epoch_loss)
            elif phase == 'val':
                val_losses.append(epoch_loss)

        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch complete in {:.0f}m {:.0f}s'.format(epoch_time_elapsed // 60, epoch_time_elapsed % 60))
        print()

    train_time_elapsed = time.time() - train_start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(train_time_elapsed // 60, train_time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print("All Train Losses:", train_losses)
    print("All Val Losses:", val_losses)

    # save model
    torch.save(best_model_wts, os.path.join(config.MODEL_DIR, '%s_%s_best.pth'%(opt.model, time_str)))

    # load best model weights and return
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # Initilization models and data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_workers = 0
    model_dict = {
    'SiameseEfficientNet': net.SiameseEfficientNet,
    'SiameseViTNet': net.SiameseViTNet,
    'SiameseResNet': net.SiameseResNet,
    'SiameseSqueezeNet': net.SiameseSqueezeNet,
    'SiameseMobileNet': net.SiameseMobileNet,
    }
    if opt.model in model_dict:
        model = model_dict[opt.model]()
    else:
        raise ValueError(f"Unsupported model: {opt.model}")
    
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1) # Decay LR by a factor of 0.1 every opt.step epochs
    criterion = net.ContrastiveLoss(margin=opt.margin)
    if opt.model == 'SiameseViTNet':
        image_datasets = {
        x: SatUAVDataset(csv_meta=f'{opt.data}.csv' if x=='train' else 'raw.csv',
                         csv_file=f'{x}.csv',
                         root_dir=config.DATA_DIR,
                         transform=None,
                         processor=ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')) for x in ['train', 'val']
        }
    else:
        image_datasets = {
        x: SatUAVDataset(csv_meta=f'{opt.data}.csv' if x=='train' else 'raw.csv',
                         csv_file=f'{x}.csv',
                         root_dir=config.DATA_DIR,
                         transform=data_transforms['norm'],
                         processor=None) for x in ['train', 'val']
        }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,
                                                  shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    print(model.__class__.__name__, 'is created at:', time_str)

    # Training
    print(model)
    model = train_model(model, dataloaders, device,
                        criterion, optimizer, lr_scheduler, time_str,
                        num_epochs=opt.nepoch)
