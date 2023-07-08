from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics.pairwise import pairwise_distances, paired_cosine_distances
import pandas as pd
import numpy as np
import os
import argparse
import time
import torch
import pickle


parser = argparse.ArgumentParser(description='Using AP clustering to discover the prototypes')
parser.add_argument('--lib', type=str, default='lib/select_train.ckpt', help='lib to save wsi id of train set')
parser.add_argument('--preference', type=int, default=0, help='preference for AP clustering')
parser.add_argument('--damping', type=int, default=0.5, help='damping for AP clustering')
parser.add_argument('--global_cluster', type=str, default='cluster/prototypes_feature.npy')
parser.add_argument('--prototypes_index', type=str, default='cluster/prototypes_index.ckpt')
parser.add_argument('--global_ap_model', type=str, default='cluster/ap_model.pkl')
parser.add_argument('--feat_dir', type=str, default='', help='path to save features')
parser.add_argument('--local_cluster', type=str, default='cluster/local_cluster.ckpt')
parser.add_argument('--lamb', type=float, default=0.25)
parser.add_argument('--feat_format', type=str, choices = ['.csv', '.npy', '.pt'], default='.csv')


global args
args = parser.parse_args()

def main():
    feature_columns = make_csv_columns() # 生成默认维度对应长度的列表，作为特征csv文件的列名
    feature_paths = get_feature_path() # 返回 feature_csv_file
    
    # intra-slide clustering (WSI-level)
    local_cluster_centroids, local_centroids_features, slideIDXs, gridIDXs = local_clustering(feature_paths, feature_columns)
    
    # inter-slide clustering (whole dataset)
    global_cluster_centroids, global_centroids_features, ap = global_clustering(local_centroids_features)

    # Saving AP clustering model
    with open(args.global_ap_model, 'wb') as f:
        pickle.dump(ap, f)

    # Saving the global prototype features
    np.savez(args.global_cluster, feature=global_centroids_features)

    # save global cluster centroids index
    save_global_centroids_idx(global_cluster_centroids, slideIDXs, gridIDXs)



def local_clustering(feature_paths, feature_columns):
    '''
    local_clustering 函数是用于执行基于单个WSI的局部聚类的部分。
    它接收特征路径列表 feature_paths 和特征列名列表 feature_columns 作为输入，并返回局部聚类的质心、WSI索引和聚类质心的索引
    '''
    local_cluster_centroids = []
    wsi_names = []
    slideIDXs = []
    cluster_gridIDXs = []
    local_centroids_features = torch.Tensor()
    
    topn_gridIDXs = get_grid_index() # 返回网格字典，key:wsi_name value:grid_indexlist
    for i, file in enumerate(feature_paths):
        wsi_name = file.split('.')[0]
        wsi_names.append(wsi_name)
        
        if args.feat_format == '.csv':
            df = pd.read_csv(os.path.join(args.feature_dir, file), index_col=0)
            feat = df[feature_columns].values
        elif args.feat_format == '.npy':
            feat = np.load(os.path.join(args.feature_dir, file))
        elif args.feat_format == '.pt':
            feat = torch.load(os.path.join(args.feature_dir, file))
        topn_idx = np.array(topn_gridIDXs[wsi_name])
        feat = feat[topn_idx, :]
        
        begin = time.time()
        similarity = euclidean_similarity(X_train) # 通过应用指数函数 np.exp(-dists) 将欧氏距离转换为相似性，得到相似性矩阵。
        ## 指数函数可以将较大的距离值映射到较小的相似性值，以便更好地表示样本之间的相似程度。
   
        # default using negative squared euclidean distance
        af = AffinityPropagation(preference=args.preference, damping=args.damping, affinity='precomputed', random_state=24).fit(similarity)
        # cluster_centers_indices = ap.cluster_centers_indices_
        cluster_centers_indices = af.cluster_centers_indices_
        n_clusters = len(cluster_centers_indices)
        '''
        fit() 方法将相似性矩阵 similarity 作为输入进行聚类
        然后，通过 cluster_centers_indices_ 属性获取聚类的质心索引，通过 len() 函数计算聚类的数量。
        '''
        end = time.time()
        local_cluster_centroids.append(cluster_centers_indices)
        usetime = end - begin
        print("wsi {}\t File name: {}\t Use time: {:.2f}\t Number of cluster: {}".format(i + 1, file, usetime,n_clusters))
        slideIDXs.extend([i] * n_clusters)
        cluster_gridIDXs.extend(cluster_centers_indices)
        # local_centroids_feature = patch_features[cluster_centers_indices, :]
        local_centroids_feature = feat[cluster_centers_indices, :]
        local_centroids_feature = local_centroids_feature.astype(np.float32)
        local_centroids_feature = torch.from_numpy(local_centroids_feature)
        local_centroids_features = torch.cat((local_centroids_features, local_centroids_feature), dim=0)
    torch.save({
        'preference': args.preference,
        'dampling': args.damping,
        'wsi_names': wsi_names,
        'centroid': local_cluster_centroids},
        args.local_cluster)
    return local_cluster_centroids, local_centroids_features, slideIDXs, cluster_gridIDXs



def global_clustering(local_centroids_features):
    local_centroids_features = np.array(local_centroids_features)
    ap = AffinityPropagation(preference=args.preference, damping=args.damping, random_state=24).fit(local_centroids_features)
    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_
    n_clusters = len(cluster_centers_indices)
    print("Estimate number of cluster: ", n_clusters)
    global_cluster_centroids = cluster_centers_indices
    global_centroids_features = local_centroids_features[global_cluster_centroids, :]
    global_centroids_features = global_centroids_features.astype(np.float32)
    return global_cluster_centroids, global_centroids_features, ap



def save_global_centroids_idx(global_cluster_centroids, slideIDXs, gridIDXs):
    '''
    函数接收三个参数：
    global_cluster_centroids：全局聚类的聚类中心索引列表。
    slideIDXs：每个聚类中心所属的幻灯片索引列表。
    gridIDXs：每个聚类中心所属的网格索引列表。

    函数通过遍历全局聚类的聚类中心索引列表，依次获取每个聚类中心的幻灯片索引和网格索引，然后将它们分别保存到 centroids_slideIDXs 和 centroids_idx 列表中。
    最后，使用 torch.save() 函数将 centroids_slideIDXs 和 centroids_idx 保存到指定的文件路径 args.prototypes_index。
    请确保在调用 save_global_centroids_idx 函数之前，已经计算得到了全局聚类的聚类中心索引、幻灯片索引和网格索引。
    '''
    centroids_slideIDXs = []
    centroids_idx = []
    for idx in global_cluster_centroids:
        slideIDX = slideIDXs[idx]
        centroids_slideIDXs.append(slideIDX)
        centroids_idx.append(gridIDXs[idx])
    torch.save({
        'slideIDX': centroids_slideIDXs,
        'gridIDX': centroids_idx},
        args.prototypes_index)


def cosine_distance(matrix1,matrix2):
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return cosine_distance
    
def euclidean_similarity(x):
    dists = pairwise_distances(x, metric='euclidean') # X 是一个形状为 (n_samples, n_features) 的数据集，表示 n_samples 个样本的特征向量
    dists =  dists / np.max(dists)
    similarity = np.exp(-dists * args.lamb)
    return similarity

def make_csv_columns(ndim=512):
    '''
    该函数接受一个参数 ndim，表示特征的维度，默认为512。
    函数通过循环从1到 ndim，生成对应的列名，命名规则为 'feature' + str(i)，将其添加到 feature_columns 列表中。最后，将生成的列名列表返回。
    '''
    feature_columns = []
    for i in range(1, ndim+1):
        feature_columns.append('feature' + str(i))
    return feature_columns

def get_feature_path():
    lib = torch.load(args.lib, map_location='cpu')
    wsi_paths = lib['slides']
    wsi_names = [os.path.basename(wsi_path).split('.')[0] for wsi_path in wsi_paths]
    feature_csv_files = [wsi_name + args.feat_format for wsi_name in wsi_names]
    # return feat_paths
    return feature_csv_files
    
def get_grid_index():
    '''
    get_grid_index 函数用于获取网格索引（grid index）的信息。
    它从预设的 args.lib 文件中获取WSI文件的路径列表和相应的网格索引，然后将它们组合成一个字典，其中键是WSI文件的名称，值是对应的网格索引列表
    '''
    gridIDXs = {}
    lib = torch.load(args.lib, map_location='cpu')
    wsi_paths = lib['slides']
    index = lib['gridIDX']
    wsi_names = [os.path.basename(wsi_path).split('.')[0] for wsi_path in wsi_paths]
    for i, wsi_name in enumerate(wsi_names):
        gridIDXs[wsi_name] = index[i]
    return gridIDXs

if __name__ == '__main__':
    main()
