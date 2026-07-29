import torch.nn.functional as F
from Dataset import *
import net
from Dataset import SatUAVDataset
import os
import time
import argparse
import config
import sys
from utils import data_transforms
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import ViTImageProcessor
import pickle
from PIL import Image

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
    # 存储匹配和不匹配对的 embedding
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

                #if not found_match and label == 0:
                 #   a1, b1 = model(A.to(device), B.to(device), return_patches=True)  # 返回 (1, N, 768)
                  #  uav_match = a1[0].squeeze(0).cpu().numpy()
                   # uav_coords=a1[1].squeeze(0).cpu().numpy()
                    #sat_match = b1[0].squeeze(0).cpu().numpy()  # shape: (N, 768)
                    #sat_coords=b1[1].squeeze(0).cpu().numpy()
                    #uav_match_img_tensor = A.squeeze(0).permute(1, 2, 0).cpu()
                    #uav_match_img = ((uav_match_img_tensor + 1) / 2 * 255).clamp(0, 255).byte().numpy()
                    #sat_match_img_tensor = B.squeeze(0).permute(1, 2, 0).cpu()
                    #sat_match_img = ((sat_match_img_tensor + 1) / 2 * 255).clamp(0, 255).byte().numpy()
                    #found_match = True
                #elif not found_nonmatch and label == 1:
                    #a2, b2 = model(A.to(device), B.to(device), return_patches=True)  # 返回 (1, N, 768)
                    #uav_nonmatch = a2[0].squeeze(0).cpu().numpy()
                    #uav_noncoords=a2[1].squeeze(0).cpu().numpy()
                    #sat_nonmatch = b2[0].squeeze(0).cpu().numpy()
                    #sat_noncoords=b2[1].squeeze(0).cpu().numpy()
                    #uav_nonmatch_img_tensor = A.squeeze(0).permute(1, 2, 0).cpu()
                    #uav_nonmatch_img = ((uav_nonmatch_img_tensor + 1) / 2 * 255).clamp(0, 255).byte().numpy()
                    #sat_nonmatch_img_tensor = B.squeeze(0).permute(1, 2, 0).cpu()
                    #sat_nonmatch_img = ((sat_nonmatch_img_tensor + 1) / 2 * 255).clamp(0, 255).byte().numpy() 
                    #found_nonmatch = True
        print()
    print((time.time()-since)/n, 'seconds/pair')

    # Draw ROC curve for test data
    fpr, tpr, thresholds = roc_curve(label_matrix['val'], output_matrix['val'])
    auc_score = roc_auc_score(label_matrix['val'], output_matrix['val'])
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"AUC score is {auc_score}.")
    print("↓↓↓↓↓↓↓↓↓ ROC data ↓↓↓↓↓↓↓↓↓↓")
    print(fpr.tolist(), ',', tpr.tolist())
    print("↑↑↑↑↑↑↑↑↑ ROC end  ↑↑↑↑↑↑↑↑↑↑")

    # generate prediction result based on threshold = 2
    #print("Prediction result based on threshold = 2 ")
    pred_matrix = {x: (output_matrix[x] > 1.7)*1 for x in ['train', 'val']}
    for x in ['train', 'val']:
        result = confusion_matrix(pred_matrix[x], label_matrix[x])
        print(f'  {x} data:')
        for k,v in zip('TP,FP,TN,FN,TPR,TNR,ACC,precision,recall,F1'.split(','), result):
            print("    ", k, ':', v)
    filename = f'vis_data_ViT.pkl'  
    with open(filename, 'wb') as f:
        pickle.dump((pred_matrix, output_matrix, label_matrix), f)
    #np.savez('embedding_data.npz', sat_match=sat_match, sat_coords=sat_coords, sat_match_img=sat_match_img, uav_match=uav_match, uav_coords=uav_coords, uav_match_img=uav_match_img, sat_nonmatch=sat_nonmatch, sat_noncoords=sat_noncoords, sat_nonmatch_img=sat_nonmatch_img, uav_nonmatch=uav_nonmatch, uav_noncoords=uav_noncoords, uav_nonmatch_img=uav_nonmatch_img)
    
def evaluate_Siamese_Error_Tolerance(model, device, threshold):
    print("Error-tolerance test starts.")
    print(f"Decision threshold: distance > {threshold:.4f} is classified as spoofed.")

    if model.__class__.__name__ == 'SiameseViTNet':
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

        def load_image(path):
            img = Image.open(path).convert('RGB')
            return processor(images=img, return_tensors="pt")['pixel_values'].to(device)
    else:
        resize_norm = data_transforms['resize_norm']

        def load_image(path):
            img = Image.open(path).convert('RGB')
            return resize_norm(img).unsqueeze(0).to(device)

    all_rows = []
    summary = {shift: [] for shift in [15, 30, 45]}
    directions = ['E', 'W', 'S', 'N']

    print("Predicting shifted satellite tiles...")

    for case_id in range(1, 6):
        print('-' * 80)
        print(f"Case {case_id}")
        raw_path = os.path.join(config.ERROR_TOLERANCE, f'{case_id}/raw/raw.jpg')
        A = load_image(raw_path)

        for shift in [15, 30, 45]:
            print(f"  {shift} meters:")
            for direction in directions:
                shifted_name = f"{direction}{shift}.jpg"
                shifted_path = os.path.join(config.ERROR_TOLERANCE, f"{case_id}/{shift}meters/{shifted_name}")
                B = load_image(shifted_path)
                with torch.set_grad_enabled(False):
                    outputs = model(A, B)
                    dist = F.pairwise_distance(outputs[0], outputs[1])
                    distance = float(dist.cpu().data.numpy()[0])
                    spoofed = int(distance > threshold)
                    summary[shift].append(distance)
                    all_rows.append((case_id, shift, direction, distance, spoofed))
                    label = "spoofed" if spoofed else "matched"
                    print(f"    {shifted_name}: distance={distance:.6f}, {label}")

    print('-' * 80)
    print("Summary by offset distance")
    print("offset_m, num_samples, mean_distance, std_distance, spoofing_detection_rate")
    for shift in [15, 30, 45]:
        distances = np.array(summary[shift])
        detection_rate = float(np.mean(distances > threshold))
        print(f"{shift}, {len(distances)}, {distances.mean():.6f}, {distances.std(ddof=0):.6f}, {detection_rate:.4f}")

    result_path = os.path.join(config.RESULT_DIR, f"error_tolerance_{model.__class__.__name__}.csv")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("case_id,offset_m,direction,distance,pred_spoofed\n")
        for row in all_rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]:.8f},{row[4]}\n")
    print(f"Saved detailed results to {result_path}")


def evaluate_Siamese_Offset(model, device, threshold, offset_root=None):
    print("New offset test starts.")
    print(f"Decision threshold: distance > {threshold:.4f} is classified as spoofed.")

    offset_root = offset_root or os.path.join(config.DATA_DIR, 'offset')
    uav_root = config.FULL_960x720

    if not os.path.isdir(offset_root):
        raise FileNotFoundError(f"Offset image root does not exist: {offset_root}")

    if model.__class__.__name__ == 'SiameseViTNet':
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

        def load_image(path):
            img = Image.open(path).convert('RGB')
            return processor(images=img, return_tensors="pt")['pixel_values'].to(device)
    else:
        resize_norm = data_transforms['resize_norm']

        def load_image(path):
            img = Image.open(path).convert('RGB')
            return resize_norm(img).unsqueeze(0).to(device)

    def find_image(case_dir, case_id, suffix):
        candidates = []
        for ext in ['jpg', 'jpeg', 'png']:
            candidates.append(os.path.join(case_dir, f"{case_id}_{suffix}.{ext}"))
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Cannot find image for {case_id}_{suffix} in {case_dir}")

    case_dirs = [
        d for d in sorted(os.listdir(offset_root))
        if os.path.isdir(os.path.join(offset_root, d))
    ]
    directions = ['E', 'W', 'S', 'N']
    offsets = [10, 50, 100]
    all_rows = []
    summary = {offset: [] for offset in offsets}

    print(f"Offset root: {offset_root}")
    print(f"Found {len(case_dirs)} cases: {', '.join(case_dirs)}")
    print("Predicting offset satellite tiles...")

    for case_id in case_dirs:
        print('-' * 80)
        print(f"Case {case_id}")
        case_dir = os.path.join(offset_root, case_id)
        uav_path = os.path.join(uav_root, f"{case_id}_B.jpg")
        if not os.path.exists(uav_path):
            raise FileNotFoundError(f"Cannot find UAV image: {uav_path}")

        B = load_image(uav_path)

        for offset in offsets:
            print(f"  {offset} meters:")
            for direction in directions:
                suffix = f"{direction}{offset:03d}"
                shifted_path = find_image(case_dir, case_id, suffix)
                A = load_image(shifted_path)
                with torch.set_grad_enabled(False):
                    outputs = model(A, B)
                    dist = F.pairwise_distance(outputs[0], outputs[1])
                    distance = float(dist.cpu().data.numpy()[0])
                    spoofed = int(distance > threshold)
                    summary[offset].append(distance)
                    all_rows.append((case_id, offset, direction, distance, spoofed))
                    print(f"    {suffix}: distance={distance:.6f}, {'spoofed' if spoofed else 'matched'}")

    print('-' * 80)
    print("Summary by offset distance")
    print("offset_m, num_samples, mean_distance, std_distance, spoofing_detection_rate")
    for offset in offsets:
        distances = np.array(summary[offset])
        detection_rate = float(np.mean(distances > threshold))
        print(f"{offset}, {len(distances)}, {distances.mean():.6f}, {distances.std(ddof=0):.6f}, {detection_rate:.4f}")

    result_path = os.path.join(config.RESULT_DIR, f"offset_{model.__class__.__name__}.csv")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("case_id,offset_m,direction,distance,pred_spoofed\n")
        for row in all_rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]:.8f},{row[4]}\n")
    print(f"Saved detailed results to {result_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_names = sorted(name for name in net.__dict__ if name.startswith("Siamese") and callable(net.__dict__[name]))
    parser.add_argument('--model', default='SiameseEfficientNet', choices=model_names, help='model architecture: ' + ' | '.join(model_names))
    parser.add_argument('--weight', type=str, help='weight file')
    parser.add_argument('--data', default='raw', choices=['raw', 'err', 'offset', 'england'])
    parser.add_argument('--margin', type=float, default=4, help='margin of Contrastive Loss, only useful in Siamese Network')
    parser.add_argument('--threshold', type=float, default=1.7, help='decision threshold for spoofing detection')
    parser.add_argument('--offset-root', type=str, default=None, help='root directory of newly collected offset images')

    opt = parser.parse_args()
    print(opt)
    sys.stdout.flush()

    if opt.model not in ['SiameseEfficientNet', 'SiameseResNet', 'SiameseSqueezeNet', 'SiameseViTNet','SiameseMobileNet']:
        quit(f"Evaluation for {opt.model} is not implemented yet.")
        # raise NotImplementedError()
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
        if opt.data == 'err':
            evaluate_Siamese_Error_Tolerance(model, device, opt.threshold)
        elif opt.data == 'offset':
            evaluate_Siamese_Offset(model, device, opt.threshold, opt.offset_root)
        else:
            evaluate_Siamese(model, device, opt.margin, opt.data)
