import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import cv2

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 18

def get_topk_matches(uav_patches, sat_patches, uav_coords, sat_coords, k=20):
    sim_matrix = cosine_similarity(uav_patches, sat_patches)  # (196, 196)

    matches = []
    for i in range(len(uav_coords)):
        j = np.argmax(sim_matrix[i]) 
        matches.append((uav_coords[i], sat_coords[j], sim_matrix[i, j]))

    # 按相似度排序，取top-k
    matches = sorted(matches, key=lambda x: x[2], reverse=True)[:k]
    return matches

def plot_patch_matches(uav_img, sat_img, matches):
    """
    uav_img, sat_img: numpy (224,224,3) RGB 图像
    matches: [(coord_uav, coord_sat, score), ...]
    """
    h, w = uav_img.shape[:2]

    gap_width = 5 

    combined = np.ones((h, w*2 + gap_width, 3), dtype=np.uint8) * 255 
    combined[:, :w, :] = uav_img
    combined[:, w+gap_width:w*2+gap_width, :] = sat_img 

    plt.figure(figsize=(10, 5))
    plt.imshow(combined)

    # 画匹配连线
    for (uav_c, sat_c, score) in matches:
        x1, y1 = uav_c
        x2, y2 = sat_c
        x2 += w + gap_width
        plt.plot([x1, x2], [y1, y2], c="#2ac3ff", linewidth=1)
        plt.scatter([x1, x2], [y1, y2], c="lime", s=10)

    plt.text(3.3, 3.3, "Satellite View", color='white', 
             fontsize=18, weight='bold', 
             ha='left', va='top',
             bbox=dict(boxstyle="square,pad=0.3", facecolor='black', alpha=0.7))
    plt.text(w + 3.3 + gap_width, 3.3, "Drone View", color='white', 
             fontsize=18, weight='bold', 
             ha='left', va='top',
             bbox=dict(boxstyle="square,pad=0.3", facecolor='black', alpha=0.7))

    plt.axis("off")
    plt.tight_layout()
    plt.savefig('2.png')

def load_image(path, size=224):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img

# 载入保存的 patch embedding
data = np.load("embedding_data.npz")
sat_match = data['sat_match'] 
sat_coords = data['sat_coords'] 
uav_match = data['uav_match']
uav_coords = data['uav_coords'] 
sat_nonmatch = data['sat_nonmatch']
sat_noncoords = data['sat_noncoords'] 
uav_nonmatch = data['uav_nonmatch']
uav_noncoords = data['uav_noncoords'] 
sat_match_img=data['sat_match_img']
uav_match_img=data['uav_match_img']
sat_nonmatch_img=data['sat_nonmatch_img']
uav_nonmatch_img=data['uav_nonmatch_img']

aerial_img = uav_match_img
satellite_img = sat_match_img
#aerial_img = uav_nonmatch_img
#satellite_img = sat_nonmatch_img

# 计算匹配
matches = get_topk_matches(uav_match, sat_match, uav_coords, sat_coords, k=20)
#nonmatches = get_topk_matches(uav_nonmatch, sat_nonmatch, uav_noncoords, sat_noncoords, k=20)

# 绘制
plot_patch_matches(aerial_img, satellite_img, matches)




