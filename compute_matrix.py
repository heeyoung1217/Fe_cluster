import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from numpy.linalg import norm
import pandas as pd
from sklearn.cluster import KMeans
from tslearn.metrics import dtw

# Euclidean
def euclidean_distance(ts_a, ts_b):
    return distance.euclidean(ts_a, ts_b)

# Manhattan
def manhattan_distance(ts_a, ts_b):
    return distance.cityblock(ts_a, ts_b)

# Cosine
def cosine_distance(ts_a, ts_b):
    return distance.cosine(ts_a, ts_b)

# Pearson
def pearson_distance(ts_a, ts_b):
    return 1 - np.corrcoef(ts_a, ts_b)[0, 1]

# Jensen-Shannon Divergence
def js_divergence(ts_a, ts_b):
    m = 0.5 * (ts_a + ts_b)
    return 0.5 * (sum(kl_div(ts_a, m)) + sum(kl_div(ts_b, m)))

# Earth Mover's Distance (Wasserstein)
def emd_distance(ts_a, ts_b):
    return wasserstein_distance(ts_a, ts_b)

# DTW
def dtw_distance(ts_a, ts_b):
    return dtw(ts_a, ts_b)

# Compute distance matrix
def compute_distance_matrix(data, dist_func):
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance_matrix[i, j] = dist_func(data[i], data[j])
    return distance_matrix



# 데이터 로드
df = pd.read_csv('dirac_con.csv')

# ID 열 제거
df = df.drop(columns='Unnamed: 0')

# 숫자로 변환 가능한지 확인하고, 불가능하면 NaN으로 변환
df = df.apply(pd.to_numeric, errors='coerce')

# NaN 값 제거 (NaN 값이 있는 행 전체 제거)
df.dropna(inplace=True)

# 이후의 클러스터링 코드
data = df.values

data = data.astype(float)



# Pearson 거리 척도를 사용하여 거리 행렬을 계산
# pearson_matrix = compute_distance_matrix(data, pearson_distance)
# print(pearson_matrix)

# euclidean 거리 척도를 사용하여 거리 행렬을 계산
# euclidean_matrix = compute_distance_matrix(data, euclidean_distance)
# print(euclidean_matrix)

# Manhattan 거리 척도를 사용하여 거리 행렬을 계산
# Manhattan_matrix = compute_distance_matrix(data, manhattan_distance)
# print(Manhattan_matrix)

# cosine 거리 척도를 사용하여 거리 행렬을 계산
# cosine_matrix = compute_distance_matrix(data, cosine_distance)
# print(cosine_matrix)

# jsd 거리 척도를 사용하여 거리 행렬을 계산
# jsd_matrix = compute_distance_matrix(data, js_divergence)
# print(jsd_matrix)

# emd 거리 척도를 사용하여 거리 행렬을 계산
# emd_matrix = compute_distance_matrix(data, emd_distance)
# print(emd_matrix)

# DTW
n_samples = data.shape[0]
distance_matrix = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        distance_matrix[i, j] = dtw_distance(data[i], data[j])


# KMeans 클러스터링
km = KMeans(n_clusters=6)  # 클러스터 수는 상황에 따라 조절 필요
labels = km.fit_predict(distance_matrix)

# 클러스터링 결과 출력
print(labels)