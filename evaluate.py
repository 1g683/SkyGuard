import torch.nn.functional as F
from Dataset import *
import net
from Dataset import SatUAVDataset
import time
import argparse
import config
import sys
from utils import data_transforms
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import ViTImageProcessor
import pickle

def confusion_matrix(pred, gt):
    assert pred.shape == gt.shape

    P = 1 
    N = 0

    TP = np.sum(np.logical_and(gt == pred, pred == P))
    TN = np.sum(np.logical_and(gt == pred, pred == N))
    FP = np.sum(np.logical_and(gt != pred, pred == P))
    FN = np.sum(np.logical_and(gt != pred, pred == N))
    
    TPR=TP/(TP+FN)
    TNR=TN/(FP+TN)
    ACC = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TPR
    F1 = 2*precision*recall/(precision+recall)
    return TP, FP, TN, FN, TPR, TNR, ACC, precision, recall, F1

def evaluate_Siamese(model, device, margin, data):
    print("Test starts.")

    # Load datasets
    print("Loading data...")
    if model.__class__.__name__ == 'SiameseViTNet':
        if data == 'raw':
            image_datasets = {
                x: SatUAVDataset(csv_meta='raw.csv', 
                                 csv_file=f'{x}.csv',
                                 root_dir=config.DATA_DIR,
                                 transform=None,
                                 processor=ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')) for x in ['train', 'val']
            }
        elif data == 'england':
            image_datasets = {
                x: SatUAVDataset(csv_meta='england.csv', 
                                 csv_file=f'england_{x}.csv',
                                 root_dir=config.DATA_DIR,
                                 transform=None,
                                 processor=ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')) for x in ['train', 'val']
            }
        else:
            print("To evaluate Siamese based networks, data must be raw or england!")
    else:
        if data == 'raw':
            image_datasets = {
                x: SatUAVDataset(csv_meta='raw.csv', 
                                 csv_file=f'{x}.csv',
                                 root_dir=config.DATA_DIR,
                                 transform=data_transforms['norm'],
                                 processor=None) for x in ['train', 'val']
            }
        elif data == 'england':
            image_datasets = {
                x: SatUAVDataset(csv_meta='england.csv', 
                                 csv_file=f'england_{x}.csv',
                                 root_dir=config.DATA_DIR,
                                 transform=data_transforms['norm'],
                                 processor=None) for x in ['train', 'val']
            }
        else:
            print("To evaluate Siamese based networks, data must be raw or england!")
    # save matched and mismatched embedding
    sat_match = None
    uav_match = None
    sat_nonmatch = None
    uav_nonmatch = None
    found_match = False
    found_nonmatch = False
    batch_size = 1
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    n = dataset_sizes['train'] + dataset_sizes['val']
    print("Data Loading: Done.")

    # Predict and get prediction matrix
    print("Predicting...")
    output_matrix = {x: np.zeros(dataset_sizes[x]) for x in ['train', 'val']}
    label_matrix = {x: np.zeros(dataset_sizes[x]) for x in ['train', 'val']}
    since = time.time()
    for phase in ['train', 'val']:
        print(phase, "phase:")
        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            print(i_batch+1, '/', dataset_sizes[phase], end='\r')
            A = sample_batched['A'].to(device)
            B = sample_batched['B'].to(device)
            # labels = sample_batched['label'].to(device)
            with torch.set_grad_enabled(False):
                outputs = model(A, B)
                dist = F.pairwise_distance(outputs[0], outputs[1])
                output_matrix[phase][i_batch] = dist.cpu().data.numpy()[0]
                label=sample_batched['label'].numpy()[0]
                label_matrix[phase][i_batch] = label

                if not found_match and label == 0:
                    sat_match, uav_match = model(A.to(device), B.to(device), return_patches=True) 
                    sat_match = sat_match.squeeze(0).cpu().numpy() 
                    uav_match = uav_match.squeeze(0).cpu().numpy()
                    found_match = True
                elif not found_nonmatch and label == 1:
                    sat_nonmatch, uav_nonmatch = model(A.to(device), B.to(device), return_patches=True)  
                    sat_nonmatch = sat_nonmatch.squeeze(0).cpu().numpy()  
                    uav_nonmatch = uav_nonmatch.squeeze(0).cpu().numpy()
                    found_nonmatch = True
        print()
    print((time.time()-since)/n, 'seconds/pair')

    # Draw ROC curve for test data
    fpr, tpr, thresholds = roc_curve(label_matrix['val'], output_matrix['val'])                                                                                                                                  
    auc_score = roc_auc_score(label_matrix['val'], output_matrix['val'])                                                                                                                                         
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)                                                                                                                                                             
    optimal_idx = np.argmin(distances)                                                                                                                                                                           
    optimal_threshold = thresholds[optimal_idx]                                                                                                                                                                  
    #print(f"best threshold: {optimal_threshold:.4f}") 
    print(f"AUC score is {auc_score}.")
    print("----------ROC data-----------")
    print(fpr.tolist(), ',', tpr.tolist())
    print("----------ROC end------------")

    pred_matrix = {x: (output_matrix[x] > 1.7)*1 for x in ['train', 'val']}
    for x in ['train', 'val']:
        result = confusion_matrix(pred_matrix[x], label_matrix[x])
        print(f'  {x} data:')
        for k,v in zip('TP,FP,TN,FN,TPR,TNR,ACC,precision,recall,F1'.split(','), result):
            print("    ", k, ':', v)
    filename = f'vis_data_ViT.pkl'  
    with open(filename, 'wb') as f:
        pickle.dump((pred_matrix, output_matrix, label_matrix), f)
    np.savez('embedding_data.npz', sat_match=sat_match, uav_match=uav_match, sat_nonmatch=sat_nonmatch, uav_nonmatch=uav_nonmatch)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_names = sorted(name for name in net.__dict__ if name.startswith("Siamese") and callable(net.__dict__[name]))
    parser.add_argument('--model', default='SiameseEfficientNet', choices=model_names, help='model architecture: ' + ' | '.join(model_names))
    parser.add_argument('--weight', type=str, help='weight file')
    parser.add_argument('--data', default='raw', choices=['raw', 'err', 'england'])
    parser.add_argument('--margin', type=float, default=4, help='margin of Contrastive Loss, only useful in Siamese Network')

    opt = parser.parse_args()
    print(opt)
    sys.stdout.flush()

    if opt.model not in ['SiameseEfficientNet', 'SiameseResNet', 'SiameseSqueezeNet', 'SiameseViTNet','SiameseMobileNet']:
        quit(f"Evaluation for {opt.model} is not implemented yet.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    sys.stdout.flush()

    model = getattr(net, opt.model)()
    model.to(device)
    model_path = os.path.join(config.MODEL_DIR, opt.weight)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded.")
    sys.stdout.flush()

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(model)
    sys.stdout.flush()

    if opt.model in ['SiameseEfficientNet', 'SiameseResNet', 'SiameseSqueezeNet', 'SiameseViTNet','SiameseMobileNet']:
        evaluate_Siamese(model, device, opt.margin, opt.data)
    else:
        quit(f"{opt.model} is not supported.")


