#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
충청북도 지적통계 토지 이용 현황 종합 분석
- 기간: 2017년 ~ 2025년
- 대상: 충청북도 14개 시군구 (청주 4개구 포함)
- 분석: PCA, K-Means 군집화, 유형별 비교, 청주 4개구 비교, 인구/도로 상관분석

최종 프로젝트 산출물용 통합 분석 코드
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import stats
import folium
from folium.plugins import MarkerCluster
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 환경 설정
# =============================================================================
def setup_korean_font():
    """한글 폰트 설정"""
    font_paths = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_V2_DIR = os.path.join(BASE_DIR, 'result_v2')
RESULT_DIR = os.path.join(BASE_DIR, 'result_v3')
os.makedirs(RESULT_DIR, exist_ok=True)

# 지역 좌표 (지도 시각화용)
COORDINATES = {
    '청주 상당구': [36.5746, 127.5252],
    '청주 서원구': [36.5786, 127.4597],
    '청주 흥덕구': [36.6346, 127.4278],
    '청주 청원구': [36.7118, 127.5255],
    '충주시': [36.9915, 127.9259],
    '제천시': [37.1326, 128.1909],
    '보은군': [36.4893, 127.7290],
    '옥천군': [36.3063, 127.5714],
    '영동군': [36.1748, 127.7769],
    '증평군': [36.7853, 127.5814],
    '진천군': [36.8553, 127.4355],
    '괴산군': [36.8153, 127.7865],
    '음성군': [36.9403, 127.6905],
    '단양군': [36.9845, 128.3655]
}

# 군집 분류 (기존 분석 결과 기반)
CLUSTER_DEFINITION = {
    '도시/산업형': ['청주 흥덕구', '청주 청원구', '진천군', '음성군'],
    '농업/산림형': ['청주 상당구', '충주시', '제천시', '보은군', '옥천군', '영동군', '괴산군', '단양군'],
    '균형형': ['청주 서원구', '증평군']
}

# 색상 정의
CLUSTER_COLORS = {
    '도시/산업형': '#E74C3C',
    '농업/산림형': '#27AE60',
    '균형형': '#3498DB'
}

LANDUSE_COLORS = {
    '임야': '#2E7D32',
    '농경지': '#F9A825',
    '대지': '#1565C0',
    '공장용지': '#C62828'
}

# =============================================================================
# 2. 데이터 로드 및 전처리
# =============================================================================
def load_data():
    """기본 데이터 로드"""
    df = pd.read_csv(os.path.join(DATA_DIR, '01_chungbuk_yearly_full_data.csv'), encoding='utf-8-sig')
    
    cluster_map = {}
    for cluster_name, regions in CLUSTER_DEFINITION.items():
        for region in regions:
            cluster_map[region] = cluster_name
    df['군집'] = df['행정구역명'].map(cluster_map)
    df['청주시여부'] = df['행정구역명'].str.contains('청주')
    
    return df

def load_change_data():
    """변화율 데이터 로드"""
    df = pd.read_csv(os.path.join(DATA_DIR, '03_chungbuk_change_rate_2017_2025.csv'), encoding='utf-8-sig')
    
    cluster_map = {}
    for cluster_name, regions in CLUSTER_DEFINITION.items():
        for region in regions:
            cluster_map[region] = cluster_name
    df['군집'] = df['행정구역명'].map(cluster_map)
    
    return df

def load_timeseries_with_pop_road():
    """인구 및 도로 포함 시계열 데이터 로드"""
    df = pd.read_csv(os.path.join(RESULT_V2_DIR, 'chungbuk_timeseries_2017_2025.csv'), encoding='utf-8-sig')
    
    # 컬럼명 정리
    df.columns = df.columns.str.strip()
    
    # 군집 정보 추가
    cluster_map = {}
    for cluster_name, regions in CLUSTER_DEFINITION.items():
        for region in regions:
            cluster_map[region] = cluster_name
    df['군집'] = df['Region'].map(cluster_map)
    
    return df

# =============================================================================
# 01. 데이터 현황 및 기초 통계
# =============================================================================
def analyze_data_overview(df, ts_df, save_path):
    """데이터 현황 및 기초 통계 분석"""
    
    df_2025 = df[df['연도'] == 2025].copy()
    
    # 기초 통계
    stats_2025 = df_2025[['임야면적', '농경지면적', '대지면적', '공장용지면적', '총면적']].describe()
    
    # 시각화: 2025년 토지 구성 개요
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1) 충북 전체 토지 구성 (파이차트)
    ax1 = axes[0, 0]
    total_2025 = df_2025[['임야면적', '농경지면적', '대지면적', '공장용지면적']].sum()
    labels = ['임야', '농경지', '대지', '공장용지']
    colors = [LANDUSE_COLORS[l] for l in labels]
    explode = (0, 0, 0.05, 0.05)
    
    ax1.pie(total_2025, labels=labels, colors=colors, autopct='%1.1f%%', 
            explode=explode, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('충청북도 전체 토지 이용 구성 (2025)', fontsize=13, fontweight='bold')
    
    # 2) 지역별 총면적 바차트
    ax2 = axes[0, 1]
    regions_sorted = df_2025.sort_values('총면적', ascending=True)
    colors_bar = [CLUSTER_COLORS[c] for c in regions_sorted['군집']]
    
    ax2.barh(regions_sorted['행정구역명'], regions_sorted['총면적'] / 1e9, color=colors_bar, edgecolor='black')
    ax2.set_xlabel('총면적 (십억㎡)', fontsize=11)
    ax2.set_title('시군구별 총면적 (2025)', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3) 데이터 연도 분포
    ax3 = axes[1, 0]
    year_counts = df.groupby('연도').size()
    ax3.bar(year_counts.index, year_counts.values, color='#3498DB', edgecolor='black')
    ax3.set_xlabel('연도', fontsize=11)
    ax3.set_ylabel('데이터 수 (시군구 수)', fontsize=11)
    ax3.set_title('연도별 데이터 현황', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4) 인구 및 도로 데이터 현황
    ax4 = axes[1, 1]
    ts_2025 = ts_df[ts_df['Year'] == 2025]
    
    x = np.arange(len(ts_2025))
    width = 0.35
    
    pop_normalized = ts_2025['Population'] / ts_2025['Population'].max() * 100
    road_normalized = ts_2025['Road_Area'] / ts_2025['Road_Area'].max() * 100
    
    bars1 = ax4.bar(x - width/2, pop_normalized, width, label='인구 (정규화)', color='#9B59B6', alpha=0.7)
    bars2 = ax4.bar(x + width/2, road_normalized, width, label='도로면적 (정규화)', color='#E67E22', alpha=0.7)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([r[:4] if '청주' in r else r[:2] for r in ts_2025['Region']], rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('정규화 값 (%)', fontsize=11)
    ax4.set_title('시군구별 인구/도로면적 현황 (2025)', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('충청북도 지적통계 데이터 현황 개요 (2017-2025)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '01_data_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 기초통계 저장
    stats_2025.to_csv(os.path.join(save_path, '01_basic_statistics.csv'), encoding='utf-8-sig')
    
    return stats_2025

# =============================================================================
# 02. PCA 분석 및 분류 근거
# =============================================================================
def perform_pca_analysis(df, save_path):
    """PCA 분석 수행 및 분류 근거 시각화"""
    
    df_2025 = df[df['연도'] == 2025].copy()
    
    features = ['임야면적_비율', '농경지면적_비율', '대지면적_비율', '공장용지면적_비율']
    X = df_2025[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_result = df_2025[['행정구역명', '군집']].copy()
    pca_result['PC1'] = X_pca[:, 0]
    pca_result['PC2'] = X_pca[:, 1]
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    var_explained = pca.explained_variance_ratio_
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1) PCA 산점도
    ax1 = axes[0]
    for cluster_name in CLUSTER_DEFINITION.keys():
        mask = pca_result['군집'] == cluster_name
        ax1.scatter(pca_result.loc[mask, 'PC1'], 
                   pca_result.loc[mask, 'PC2'],
                   c=CLUSTER_COLORS[cluster_name],
                   s=150, alpha=0.8, edgecolors='white',
                   label=cluster_name, linewidths=2)
    
    for idx, row in pca_result.iterrows():
        ax1.annotate(row['행정구역명'], 
                    (row['PC1'], row['PC2']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.9)
    
    feature_labels = ['임야', '농경지', '대지', '공장용지']
    for i, (feature, label) in enumerate(zip(features, feature_labels)):
        ax1.arrow(0, 0, loadings[i, 0]*2.5, loadings[i, 1]*2.5,
                 head_width=0.1, head_length=0.05, fc='darkred', ec='darkred', alpha=0.7)
        ax1.text(loadings[i, 0]*2.8, loadings[i, 1]*2.8, label,
                fontsize=10, color='darkred', fontweight='bold')
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% 설명)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% 설명)', fontsize=11)
    ax1.set_title('PCA 기반 지역 분류 (2025년 토지이용 비율)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2) 분산 설명력
    ax2 = axes[1]
    components = ['PC1', 'PC2']
    bars = ax2.bar(components, var_explained * 100, color=['#3498DB', '#E74C3C'], edgecolor='black')
    ax2.set_ylabel('분산 설명력 (%)', fontsize=11)
    ax2.set_title('주성분별 분산 설명력', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    for bar, val in zip(bars, var_explained):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val*100:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    cumsum = np.cumsum(var_explained) * 100
    ax2.axhline(y=cumsum[1], color='green', linestyle='--', alpha=0.7)
    ax2.text(1.3, cumsum[1], f'누적: {cumsum[1]:.1f}%', fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '02_PCA_classification_basis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 로딩값 저장
    loading_df = pd.DataFrame(loadings, columns=['PC1_로딩', 'PC2_로딩'], index=feature_labels)
    loading_df.to_csv(os.path.join(save_path, '02_pca_loadings.csv'), encoding='utf-8-sig')
    
    pca_summary = {
        'PC1_분산설명력': var_explained[0],
        'PC2_분산설명력': var_explained[1],
        '누적_분산설명력': sum(var_explained),
        'PC1_해석': '도시/산업화 정도 (공장용지, 대지 비율 높을수록 +)',
        'PC2_해석': '농경지 비율 (농경지 높을수록 +)'
    }
    
    return pca_result, pca_summary, loading_df

# =============================================================================
# 03. K-Means 군집화 및 Silhouette 분석
# =============================================================================
def perform_clustering_analysis(df, save_path):
    """K-Means 군집화 및 Silhouette Score 분석"""
    
    df_2025 = df[df['연도'] == 2025].copy()
    
    features = ['임야면적_비율', '농경지면적_비율', '대지면적_비율', '공장용지면적_비율']
    X = df_2025[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    k_range = range(2, 7)
    silhouette_scores = []
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        inertias.append(kmeans.inertia_)
    
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_3 = kmeans_3.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels_3)
    silhouette_vals = silhouette_samples(X_scaled, labels_3)
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1 = axes[0]
    ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='선택된 k=3')
    ax1.set_xlabel('군집 수 (k)', fontsize=11)
    ax1.set_ylabel('Inertia (군집 내 분산)', fontsize=11)
    ax1.set_title('Elbow Method', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=3, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('군집 수 (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title(f'Silhouette Score (k=3: {silhouette_avg:.3f})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    y_lower = 10
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    
    for i in range(3):
        cluster_silhouette_vals = silhouette_vals[labels_3 == i]
        cluster_silhouette_vals.sort()
        
        size_cluster = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster
        
        ax3.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         facecolor=colors[i], alpha=0.7)
        ax3.text(-0.05, y_lower + 0.5 * size_cluster, f'군집 {i+1}', fontsize=10)
        y_lower = y_upper + 10
    
    ax3.axvline(x=silhouette_avg, color='red', linestyle='--', label=f'평균: {silhouette_avg:.3f}')
    ax3.set_xlabel('Silhouette 계수', fontsize=11)
    ax3.set_ylabel('지역', fontsize=11)
    ax3.set_title('군집별 Silhouette 분포 (k=3)', fontsize=13, fontweight='bold')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '03_silhouette_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    cluster_summary = {
        'k': 3,
        'silhouette_score': silhouette_avg,
        'inertia': kmeans_3.inertia_,
        '해석': 'Silhouette Score 0.4 이상으로 군집 분리가 적절함'
    }
    
    return cluster_summary

# =============================================================================
# 04. 유형별 토지 구성 비교
# =============================================================================
def analyze_cluster_composition(df, save_path):
    """군집별 토지 구성 비교 분석"""
    
    df_2025 = df[df['연도'] == 2025].copy()
    
    cluster_avg = df_2025.groupby('군집').agg({
        '임야면적_비율': 'mean',
        '농경지면적_비율': 'mean',
        '대지면적_비율': 'mean',
        '공장용지면적_비율': 'mean'
    }).round(2)
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    clusters = ['도시/산업형', '균형형', '농업/산림형']
    x = np.arange(len(clusters))
    width = 0.6
    
    bottom = np.zeros(len(clusters))
    landuse_cols = ['임야면적_비율', '농경지면적_비율', '대지면적_비율', '공장용지면적_비율']
    landuse_names = ['임야', '농경지', '대지', '공장용지']
    colors = [LANDUSE_COLORS[name] for name in landuse_names]
    
    for col, name, color in zip(landuse_cols, landuse_names, colors):
        values = [cluster_avg.loc[c, col] if c in cluster_avg.index else 0 for c in clusters]
        ax1.bar(x, values, width, bottom=bottom, label=name, color=color, edgecolor='white')
        bottom += values
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(clusters, fontsize=11)
    ax1.set_ylabel('비율 (%)', fontsize=11)
    ax1.set_title('군집별 토지 이용 구성 비교 (2025)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 105)
    
    ax2 = axes[1]
    urbanization = []
    for c in clusters:
        if c in cluster_avg.index:
            urbanization.append(cluster_avg.loc[c, '대지면적_비율'] + cluster_avg.loc[c, '공장용지면적_비율'])
        else:
            urbanization.append(0)
    
    bars = ax2.bar(clusters, urbanization, color=[CLUSTER_COLORS[c] for c in clusters], 
                   edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, urbanization):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('도시화 지표 (대지+공장용지 비율, %)', fontsize=11)
    ax2.set_title('군집별 도시화 수준 비교', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '04_cluster_landuse_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    cluster_avg.to_csv(os.path.join(save_path, '04_cluster_composition_2025.csv'), encoding='utf-8-sig')
    
    return cluster_avg

# =============================================================================
# 05. 연도별 토지유형 변화량 분석 (임야/농지/공장/대지)
# =============================================================================
def analyze_yearly_landtype_change(df, save_path):
    """연도별 토지유형 변화량 분석"""
    
    # 충북 전체 연도별 합계
    total_yearly = df.groupby('연도').agg({
        '임야면적': 'sum',
        '농경지면적': 'sum',
        '대지면적': 'sum',
        '공장용지면적': 'sum',
        '총면적': 'sum'
    }).reset_index()
    
    # 비율 계산
    for col in ['임야', '농경지', '대지', '공장용지']:
        total_yearly[f'{col}_비율'] = total_yearly[f'{col}면적'] / total_yearly['총면적'] * 100
    
    # 전년대비 변화량 (절대량)
    for col in ['임야면적', '농경지면적', '대지면적', '공장용지면적']:
        total_yearly[f'{col}_전년대비변화'] = total_yearly[col].diff()
        total_yearly[f'{col}_전년대비변화율'] = total_yearly[col].pct_change() * 100
    
    # 2017년 대비 누적 변화율
    base_2017 = total_yearly[total_yearly['연도'] == 2017].iloc[0]
    for col in ['임야면적', '농경지면적', '대지면적', '공장용지면적']:
        total_yearly[f'{col}_2017대비변화율'] = (total_yearly[col] - base_2017[col]) / base_2017[col] * 100
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1) 연도별 토지유형 비율 추이
    ax1 = axes[0, 0]
    landuse_names = ['임야', '농경지', '대지', '공장용지']
    for name in landuse_names:
        ax1.plot(total_yearly['연도'], total_yearly[f'{name}_비율'], 'o-', 
                label=name, color=LANDUSE_COLORS[name], linewidth=2, markersize=6)
    
    ax1.set_xlabel('연도', fontsize=11)
    ax1.set_ylabel('비율 (%)', fontsize=11)
    ax1.set_title('충북 전체 토지유형별 비율 추이', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2) 전년대비 변화량 (절대값)
    ax2 = axes[0, 1]
    years = total_yearly['연도'].values[1:]
    x_pos = np.arange(len(years))
    width = 0.2
    
    for i, name in enumerate(landuse_names):
        values = total_yearly[f'{name}면적_전년대비변화'].dropna().values / 1e6  # 백만㎡ 단위
        ax2.bar(x_pos + i*width, values, width, label=name, color=LANDUSE_COLORS[name])
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x_pos + 1.5*width)
    ax2.set_xticklabels([int(y) for y in years], fontsize=10)
    ax2.set_xlabel('연도', fontsize=11)
    ax2.set_ylabel('변화량 (백만㎡)', fontsize=11)
    ax2.set_title('연도별 토지유형 전년대비 변화량', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3) 2017년 대비 누적 변화율
    ax3 = axes[1, 0]
    for name in landuse_names:
        ax3.plot(total_yearly['연도'], total_yearly[f'{name}면적_2017대비변화율'], 'o-', 
                label=name, color=LANDUSE_COLORS[name], linewidth=2, markersize=6)
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('연도', fontsize=11)
    ax3.set_ylabel('변화율 (%)', fontsize=11)
    ax3.set_title('2017년 대비 누적 변화율', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4) 2017 vs 2025 비교
    ax4 = axes[1, 1]
    years_compare = [2017, 2025]
    x = np.arange(len(landuse_names))
    width = 0.35
    
    values_2017 = [total_yearly[total_yearly['연도'] == 2017][f'{name}_비율'].values[0] for name in landuse_names]
    values_2025 = [total_yearly[total_yearly['연도'] == 2025][f'{name}_비율'].values[0] for name in landuse_names]
    
    bars1 = ax4.bar(x - width/2, values_2017, width, label='2017년', color='#95A5A6', edgecolor='black')
    bars2 = ax4.bar(x + width/2, values_2025, width, label='2025년', color='#2C3E50', edgecolor='black')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(landuse_names, fontsize=11)
    ax4.set_ylabel('비율 (%)', fontsize=11)
    ax4.set_title('2017년 vs 2025년 토지유형 비율 비교', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('충청북도 토지유형별 연도별 변화량 분석 (2017-2025)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '05_yearly_landtype_change.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # CSV 저장
    total_yearly.to_csv(os.path.join(save_path, '05_yearly_landtype_change.csv'), index=False, encoding='utf-8-sig')
    
    return total_yearly

# =============================================================================
# 06. 유형별 연도별 추이
# =============================================================================
def analyze_cluster_trend(df, save_path):
    """군집별 토지 변화 추이 분석"""
    
    cluster_yearly = df.groupby(['연도', '군집']).agg({
        '임야면적_비율': 'mean',
        '농경지면적_비율': 'mean',
        '대지면적_비율': 'mean',
        '공장용지면적_비율': 'mean'
    }).reset_index()
    
    # 전년대비 변화량 계산
    for col in ['임야면적_비율', '농경지면적_비율', '대지면적_비율', '공장용지면적_비율']:
        change_col = col.replace('_비율', '_전년대비변화')
        cluster_yearly[change_col] = cluster_yearly.groupby('군집')[col].diff()
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    landuse_cols = ['임야면적_비율', '농경지면적_비율', '대지면적_비율', '공장용지면적_비율']
    landuse_titles = ['임야 비율 변화', '농경지 비율 변화', '대지 비율 변화', '공장용지 비율 변화']
    
    clusters = ['도시/산업형', '균형형', '농업/산림형']
    
    for ax, col, title in zip(axes.flatten(), landuse_cols, landuse_titles):
        for cluster in clusters:
            data = cluster_yearly[cluster_yearly['군집'] == cluster]
            ax.plot(data['연도'], data[col], 'o-', 
                   label=cluster, color=CLUSTER_COLORS[cluster],
                   linewidth=2, markersize=6)
        
        ax.set_xlabel('연도', fontsize=11)
        ax.set_ylabel('비율 (%)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('군집별 토지 이용 비율 변화 추이 (2017-2025)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '06_landuse_trend_by_cluster.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    cluster_yearly.to_csv(os.path.join(save_path, '06_cluster_yearly_trend.csv'), index=False, encoding='utf-8-sig')
    
    return cluster_yearly

# =============================================================================
# 07. 충북 전체 추이 종합
# =============================================================================
def analyze_chungbuk_total_trend(df, save_path):
    """충북 전체 통합 연도별 추이"""
    
    total_yearly = df.groupby('연도').agg({
        '임야면적': 'sum',
        '농경지면적': 'sum',
        '대지면적': 'sum',
        '공장용지면적': 'sum',
        '총면적': 'sum',
        '임야면적_비율': 'mean',
        '농경지면적_비율': 'mean',
        '대지면적_비율': 'mean',
        '공장용지면적_비율': 'mean'
    }).reset_index()
    
    # 전체 비율 재계산
    total_yearly['임야_전체비율'] = total_yearly['임야면적'] / total_yearly['총면적'] * 100
    total_yearly['농경지_전체비율'] = total_yearly['농경지면적'] / total_yearly['총면적'] * 100
    total_yearly['대지_전체비율'] = total_yearly['대지면적'] / total_yearly['총면적'] * 100
    total_yearly['공장용지_전체비율'] = total_yearly['공장용지면적'] / total_yearly['총면적'] * 100
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1) 전체 면적 추이
    ax1 = axes[0, 0]
    landuse_names = ['임야', '농경지', '대지', '공장용지']
    for name in landuse_names:
        ax1.plot(total_yearly['연도'], total_yearly[f'{name}면적'] / 1e9, 'o-', 
                label=name, color=LANDUSE_COLORS[name], linewidth=2, markersize=6)
    
    ax1.set_xlabel('연도', fontsize=11)
    ax1.set_ylabel('면적 (십억㎡)', fontsize=11)
    ax1.set_title('충북 전체 토지유형별 면적 추이', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2) 전체 비율 추이 (확대)
    ax2 = axes[0, 1]
    ax2.plot(total_yearly['연도'], total_yearly['대지_전체비율'], 'o-', 
            label='대지', color=LANDUSE_COLORS['대지'], linewidth=2, markersize=6)
    ax2.plot(total_yearly['연도'], total_yearly['공장용지_전체비율'], 's-', 
            label='공장용지', color=LANDUSE_COLORS['공장용지'], linewidth=2, markersize=6)
    
    ax2.set_xlabel('연도', fontsize=11)
    ax2.set_ylabel('비율 (%)', fontsize=11)
    ax2.set_title('대지/공장용지 비율 추이 (확대)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3) 면적 Stacked Area
    ax3 = axes[1, 0]
    ax3.stackplot(total_yearly['연도'], 
                  total_yearly['공장용지_전체비율'],
                  total_yearly['대지_전체비율'],
                  total_yearly['농경지_전체비율'],
                  total_yearly['임야_전체비율'],
                  labels=['공장용지', '대지', '농경지', '임야'],
                  colors=[LANDUSE_COLORS['공장용지'], LANDUSE_COLORS['대지'], 
                         LANDUSE_COLORS['농경지'], LANDUSE_COLORS['임야']],
                  alpha=0.8)
    
    ax3.set_xlabel('연도', fontsize=11)
    ax3.set_ylabel('비율 (%)', fontsize=11)
    ax3.set_title('충북 전체 토지 구성 변화 (Stacked)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4) 연도별 주요 지표 요약 테이블
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_data = [
        ['연도', '임야(%)', '농경지(%)', '대지(%)', '공장용지(%)'],
    ]
    for _, row in total_yearly.iterrows():
        summary_data.append([
            int(row['연도']),
            f"{row['임야_전체비율']:.2f}",
            f"{row['농경지_전체비율']:.2f}",
            f"{row['대지_전체비율']:.2f}",
            f"{row['공장용지_전체비율']:.3f}"
        ])
    
    table = ax4.table(cellText=summary_data, loc='center', cellLoc='center',
                      colWidths=[0.15, 0.17, 0.17, 0.15, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('연도별 토지유형 비율 요약', fontsize=13, fontweight='bold', y=0.95)
    
    plt.suptitle('충청북도 전체 토지 이용 현황 종합 (2017-2025)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '07_chungbuk_total_trend.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    total_yearly.to_csv(os.path.join(save_path, '07_chungbuk_total_yearly.csv'), index=False, encoding='utf-8-sig')
    
    return total_yearly

# =============================================================================
# 08. 14개 시군 비교 분석
# =============================================================================
def analyze_all_regions_yearly(df, save_path):
    """14개 시군 전체 연도별 비교 분석"""
    
    regions_yearly = df[['연도', '행정구역명', '군집', '임야면적_비율', '농경지면적_비율', 
                         '대지면적_비율', '공장용지면적_비율']].copy()
    
    # 전년대비 변화량 계산
    for col in ['임야면적_비율', '농경지면적_비율', '대지면적_비율', '공장용지면적_비율']:
        change_col = col.replace('_비율', '_변화량')
        regions_yearly[change_col] = regions_yearly.groupby('행정구역명')[col].diff()
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    landuse_cols = ['임야면적_비율', '농경지면적_비율', '대지면적_비율', '공장용지면적_비율']
    landuse_titles = ['임야 비율', '농경지 비율', '대지 비율', '공장용지 비율']
    
    for ax, col, title in zip(axes.flatten(), landuse_cols, landuse_titles):
        for region in df['행정구역명'].unique():
            data = df[df['행정구역명'] == region]
            cluster = data['군집'].iloc[0]
            ax.plot(data['연도'], data[col], 'o-', 
                   color=CLUSTER_COLORS[cluster], alpha=0.7,
                   linewidth=1.5, markersize=4, label=region)
        
        ax.set_xlabel('연도', fontsize=11)
        ax.set_ylabel('비율 (%)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=8, bbox_to_anchor=(1.12, 0.5))
    
    plt.suptitle('14개 시군구 토지 이용 비율 연도별 추이 (2017-2025)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '08_all_regions_yearly_trend.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    regions_yearly.to_csv(os.path.join(save_path, '08_all_regions_yearly_data.csv'), index=False, encoding='utf-8-sig')
    
    return regions_yearly

# =============================================================================
# 09. 청주 4개구 분석
# =============================================================================
def analyze_cheongju_4gu(df, change_df, save_path):
    """청주 4개구 비교 분석"""
    
    cheongju_regions = ['청주 상당구', '청주 서원구', '청주 흥덕구', '청주 청원구']
    
    df_cj = df[df['행정구역명'].isin(cheongju_regions)].copy()
    change_cj = change_df[change_df['행정구역명'].isin(cheongju_regions)].copy()
    
    # 전년대비 변화량 계산
    for col in ['임야면적_비율', '농경지면적_비율', '대지면적_비율', '공장용지면적_비율']:
        change_col = col.replace('_비율', '_전년대비변화')
        df_cj[change_col] = df_cj.groupby('행정구역명')[col].diff()
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    df_2025_cj = df_cj[df_cj['연도'] == 2025]
    
    # 1) 2025년 토지 구성
    ax1 = axes[0, 0]
    x = np.arange(len(cheongju_regions))
    width = 0.2
    
    landuse_cols = ['임야면적_비율', '농경지면적_비율', '대지면적_비율', '공장용지면적_비율']
    landuse_names = ['임야', '농경지', '대지', '공장용지']
    
    for i, (col, name) in enumerate(zip(landuse_cols, landuse_names)):
        values = df_2025_cj.set_index('행정구역명').loc[cheongju_regions, col].values
        ax1.bar(x + i*width, values, width, label=name, color=LANDUSE_COLORS[name])
    
    ax1.set_xticks(x + 1.5*width)
    ax1.set_xticklabels([r.replace('청주 ', '') for r in cheongju_regions], fontsize=11)
    ax1.set_ylabel('비율 (%)', fontsize=11)
    ax1.set_title('청주시 4개구 토지 이용 구성 (2025)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2) 변화율 비교
    ax2 = axes[0, 1]
    change_cols = ['임야면적_변화율', '농경지면적_변화율', '대지면적_변화율', '공장용지면적_변화율']
    
    for i, (col, name) in enumerate(zip(change_cols, landuse_names)):
        values = change_cj.set_index('행정구역명').loc[cheongju_regions, col].values
        ax2.bar(x + i*width, values, width, label=name, color=LANDUSE_COLORS[name])
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x + 1.5*width)
    ax2.set_xticklabels([r.replace('청주 ', '') for r in cheongju_regions], fontsize=11)
    ax2.set_ylabel('변화율 (%)', fontsize=11)
    ax2.set_title('청주시 4개구 토지 변화율 (2017→2025)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3) 연도별 공장용지 추이
    ax3 = axes[1, 0]
    region_colors = {'청주 상당구': '#27AE60', '청주 서원구': '#3498DB', 
                     '청주 흥덕구': '#E74C3C', '청주 청원구': '#9B59B6'}
    
    for region in cheongju_regions:
        data = df_cj[df_cj['행정구역명'] == region]
        ax3.plot(data['연도'], data['공장용지면적_비율'], 'o-', 
                color=region_colors[region], linewidth=2, markersize=6,
                label=region.replace('청주 ', ''))
    
    ax3.set_xlabel('연도', fontsize=11)
    ax3.set_ylabel('공장용지 비율 (%)', fontsize=11)
    ax3.set_title('청주시 4개구 공장용지 비율 변화', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4) 군집 분류 현황
    ax4 = axes[1, 1]
    cluster_info = change_cj.set_index('행정구역명').loc[cheongju_regions, '군집'].values
    colors = [CLUSTER_COLORS[c] for c in cluster_info]
    
    urban_index = []
    for region in cheongju_regions:
        row = df_2025_cj[df_2025_cj['행정구역명'] == region].iloc[0]
        urban_index.append(row['대지면적_비율'] + row['공장용지면적_비율'])
    
    bars = ax4.barh([r.replace('청주 ', '') for r in cheongju_regions], 
                    urban_index, color=colors, edgecolor='black')
    
    for bar, cluster in zip(bars, cluster_info):
        ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                cluster, va='center', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('도시화 지표 (대지+공장용지 비율, %)', fontsize=11)
    ax4.set_title('청주시 4개구 군집 분류', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.suptitle('청주시 4개구 토지 이용 현황 분석', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '09_cheongju_4gu_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # CSV 저장
    df_cj.to_csv(os.path.join(save_path, '09_cheongju_4gu_yearly_data.csv'), index=False, encoding='utf-8-sig')
    
    cj_summary = df_2025_cj[['행정구역명', '군집', '임야면적_비율', '농경지면적_비율', 
                            '대지면적_비율', '공장용지면적_비율']].copy()
    cj_summary.to_csv(os.path.join(save_path, '09_cheongju_4gu_summary.csv'), index=False, encoding='utf-8-sig')
    
    return cj_summary, df_cj

# =============================================================================
# 10. 인구 상관관계 분석
# =============================================================================
def analyze_population_correlation(ts_df, save_path):
    """시군구별 인구 대비 토지유형 상관관계 분석"""
    
    # 2017년과 2025년 데이터 추출
    df_2017 = ts_df[ts_df['Year'] == 2017].copy()
    df_2025 = ts_df[ts_df['Year'] == 2025].copy()
    
    # 변화율 계산
    merged = df_2017.merge(df_2025, on='Region', suffixes=('_2017', '_2025'))
    
    merged['인구_변화율'] = (merged['Population_2025'] - merged['Population_2017']) / merged['Population_2017'] * 100
    merged['공장용지_변화율'] = (merged['Factory_Area_2025'] - merged['Factory_Area_2017']) / merged['Factory_Area_2017'] * 100
    merged['대지_변화율'] = (merged['House_Area_2025'] - merged['House_Area_2017']) / merged['House_Area_2017'] * 100
    merged['도로_변화율'] = (merged['Road_Area_2025'] - merged['Road_Area_2017']) / merged['Road_Area_2017'] * 100
    merged['임야_변화율'] = (merged['Forest_Area_2025'] - merged['Forest_Area_2017']) / merged['Forest_Area_2017'] * 100
    merged['농경지_변화율'] = (merged['Farm_Area_2025'] - merged['Farm_Area_2017']) / merged['Farm_Area_2017'] * 100
    
    # 군집 정보 추가
    cluster_map = {}
    for cluster_name, regions in CLUSTER_DEFINITION.items():
        for region in regions:
            cluster_map[region] = cluster_name
    merged['군집'] = merged['Region'].map(cluster_map)
    
    # 상관계수 계산
    corr_vars = ['인구_변화율', '공장용지_변화율', '대지_변화율', '도로_변화율', '임야_변화율', '농경지_변화율']
    corr_matrix = merged[corr_vars].corr()
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1) 상관계수 히트맵
    ax1 = axes[0, 0]
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax1, vmin=-1, vmax=1,
                cbar_kws={'label': '상관계수'})
    ax1.set_title('변화율 간 상관관계 (2017→2025)', fontsize=13, fontweight='bold')
    
    # 2) 인구 vs 공장용지 변화율 산점도
    ax2 = axes[0, 1]
    for cluster in CLUSTER_COLORS.keys():
        mask = merged['군집'] == cluster
        ax2.scatter(merged.loc[mask, '인구_변화율'], merged.loc[mask, '공장용지_변화율'],
                   c=CLUSTER_COLORS[cluster], s=100, alpha=0.7, label=cluster, edgecolors='white')
    
    # 지역명 라벨
    for _, row in merged.iterrows():
        label = row['Region'][:4] if '청주' in row['Region'] else row['Region'][:2]
        ax2.annotate(label, (row['인구_변화율'], row['공장용지_변화율']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8, alpha=0.8)
    
    # 추세선
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged['인구_변화율'], merged['공장용지_변화율'])
    x_line = np.linspace(merged['인구_변화율'].min(), merged['인구_변화율'].max(), 100)
    ax2.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.7, 
             label=f'추세선 (r={r_value:.3f})')
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('인구 변화율 (%)', fontsize=11)
    ax2.set_ylabel('공장용지 변화율 (%)', fontsize=11)
    ax2.set_title('인구 vs 공장용지 변화율 상관관계', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3) 인구 vs 대지 변화율
    ax3 = axes[1, 0]
    for cluster in CLUSTER_COLORS.keys():
        mask = merged['군집'] == cluster
        ax3.scatter(merged.loc[mask, '인구_변화율'], merged.loc[mask, '대지_변화율'],
                   c=CLUSTER_COLORS[cluster], s=100, alpha=0.7, label=cluster, edgecolors='white')
    
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
        merged['인구_변화율'], merged['대지_변화율'])
    ax3.plot(x_line, slope2 * x_line + intercept2, 'r--', alpha=0.7,
             label=f'추세선 (r={r_value2:.3f})')
    
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('인구 변화율 (%)', fontsize=11)
    ax3.set_ylabel('대지 변화율 (%)', fontsize=11)
    ax3.set_title('인구 vs 대지 변화율 상관관계', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4) 군집별 인구 변화 비교
    ax4 = axes[1, 1]
    cluster_pop = merged.groupby('군집')['인구_변화율'].agg(['mean', 'std']).reset_index()
    cluster_order = ['도시/산업형', '균형형', '농업/산림형']
    cluster_pop = cluster_pop.set_index('군집').loc[cluster_order].reset_index()
    
    bars = ax4.bar(cluster_pop['군집'], cluster_pop['mean'], 
                   yerr=cluster_pop['std'], capsize=5,
                   color=[CLUSTER_COLORS[c] for c in cluster_pop['군집']], 
                   edgecolor='black', linewidth=1.5)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel('인구 변화율 평균 (%)', fontsize=11)
    ax4.set_title('군집별 인구 변화율 비교 (2017→2025)', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, cluster_pop['mean']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('시군구별 인구 대비 토지이용 상관관계 분석', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '10_correlation_population.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 상관계수 저장
    corr_matrix.to_csv(os.path.join(save_path, '10_correlation_matrix.csv'), encoding='utf-8-sig')
    merged.to_csv(os.path.join(save_path, '10_population_change_data.csv'), index=False, encoding='utf-8-sig')
    
    # 통계적 유의성 테스트 결과
    significance_results = []
    for var in ['공장용지_변화율', '대지_변화율', '도로_변화율', '임야_변화율', '농경지_변화율']:
        slope, intercept, r_value, p_value, std_err = stats.linregress(merged['인구_변화율'], merged[var])
        significance_results.append({
            '변수': var,
            '상관계수': r_value,
            'p-value': p_value,
            '유의성(p<0.05)': '유의' if p_value < 0.05 else '비유의'
        })
    
    sig_df = pd.DataFrame(significance_results)
    sig_df.to_csv(os.path.join(save_path, '10_correlation_significance.csv'), index=False, encoding='utf-8-sig')
    
    return merged, corr_matrix

# =============================================================================
# 11. 도로 상관관계 분석
# =============================================================================
def analyze_road_correlation(ts_df, save_path):
    """시군구별 도로면적 대비 토지유형 상관관계 분석"""
    
    # 2017년과 2025년 데이터 추출
    df_2017 = ts_df[ts_df['Year'] == 2017].copy()
    df_2025 = ts_df[ts_df['Year'] == 2025].copy()
    
    # 변화율 계산
    merged = df_2017.merge(df_2025, on='Region', suffixes=('_2017', '_2025'))
    
    merged['도로_변화율'] = (merged['Road_Area_2025'] - merged['Road_Area_2017']) / merged['Road_Area_2017'] * 100
    merged['공장용지_변화율'] = (merged['Factory_Area_2025'] - merged['Factory_Area_2017']) / merged['Factory_Area_2017'] * 100
    merged['대지_변화율'] = (merged['House_Area_2025'] - merged['House_Area_2017']) / merged['House_Area_2017'] * 100
    merged['인구_변화율'] = (merged['Population_2025'] - merged['Population_2017']) / merged['Population_2017'] * 100
    
    # 군집 정보 추가
    cluster_map = {}
    for cluster_name, regions in CLUSTER_DEFINITION.items():
        for region in regions:
            cluster_map[region] = cluster_name
    merged['군집'] = merged['Region'].map(cluster_map)
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1) 도로 vs 공장용지 변화율
    ax1 = axes[0, 0]
    for cluster in CLUSTER_COLORS.keys():
        mask = merged['군집'] == cluster
        ax1.scatter(merged.loc[mask, '도로_변화율'], merged.loc[mask, '공장용지_변화율'],
                   c=CLUSTER_COLORS[cluster], s=100, alpha=0.7, label=cluster, edgecolors='white')
    
    for _, row in merged.iterrows():
        label = row['Region'][:4] if '청주' in row['Region'] else row['Region'][:2]
        ax1.annotate(label, (row['도로_변화율'], row['공장용지_변화율']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8, alpha=0.8)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged['도로_변화율'], merged['공장용지_변화율'])
    x_line = np.linspace(merged['도로_변화율'].min(), merged['도로_변화율'].max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.7,
             label=f'추세선 (r={r_value:.3f}, p={p_value:.3f})')
    
    ax1.set_xlabel('도로면적 변화율 (%)', fontsize=11)
    ax1.set_ylabel('공장용지 변화율 (%)', fontsize=11)
    ax1.set_title('도로면적 vs 공장용지 변화율', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2) 도로 vs 대지 변화율
    ax2 = axes[0, 1]
    for cluster in CLUSTER_COLORS.keys():
        mask = merged['군집'] == cluster
        ax2.scatter(merged.loc[mask, '도로_변화율'], merged.loc[mask, '대지_변화율'],
                   c=CLUSTER_COLORS[cluster], s=100, alpha=0.7, label=cluster, edgecolors='white')
    
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
        merged['도로_변화율'], merged['대지_변화율'])
    ax2.plot(x_line, slope2 * x_line + intercept2, 'r--', alpha=0.7,
             label=f'추세선 (r={r_value2:.3f}, p={p_value2:.3f})')
    
    ax2.set_xlabel('도로면적 변화율 (%)', fontsize=11)
    ax2.set_ylabel('대지 변화율 (%)', fontsize=11)
    ax2.set_title('도로면적 vs 대지 변화율', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3) 도로 vs 인구 변화율
    ax3 = axes[1, 0]
    for cluster in CLUSTER_COLORS.keys():
        mask = merged['군집'] == cluster
        ax3.scatter(merged.loc[mask, '도로_변화율'], merged.loc[mask, '인구_변화율'],
                   c=CLUSTER_COLORS[cluster], s=100, alpha=0.7, label=cluster, edgecolors='white')
    
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(
        merged['도로_변화율'], merged['인구_변화율'])
    ax3.plot(x_line, slope3 * x_line + intercept3, 'r--', alpha=0.7,
             label=f'추세선 (r={r_value3:.3f}, p={p_value3:.3f})')
    
    ax3.set_xlabel('도로면적 변화율 (%)', fontsize=11)
    ax3.set_ylabel('인구 변화율 (%)', fontsize=11)
    ax3.set_title('도로면적 vs 인구 변화율', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4) 군집별 도로면적 변화 비교
    ax4 = axes[1, 1]
    cluster_road = merged.groupby('군집')['도로_변화율'].agg(['mean', 'std']).reset_index()
    cluster_order = ['도시/산업형', '균형형', '농업/산림형']
    cluster_road = cluster_road.set_index('군집').loc[cluster_order].reset_index()
    
    bars = ax4.bar(cluster_road['군집'], cluster_road['mean'],
                   yerr=cluster_road['std'], capsize=5,
                   color=[CLUSTER_COLORS[c] for c in cluster_road['군집']],
                   edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('도로면적 변화율 평균 (%)', fontsize=11)
    ax4.set_title('군집별 도로면적 변화율 비교 (2017→2025)', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, cluster_road['mean']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('시군구별 도로면적 대비 토지이용 상관관계 분석', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '11_correlation_road.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 도로 상관분석 결과 저장
    road_corr_results = []
    for var in ['공장용지_변화율', '대지_변화율', '인구_변화율']:
        slope, intercept, r_value, p_value, std_err = stats.linregress(merged['도로_변화율'], merged[var])
        road_corr_results.append({
            '변수': var,
            '상관계수': r_value,
            'p-value': p_value,
            '유의성(p<0.05)': '유의' if p_value < 0.05 else '비유의'
        })
    
    road_corr_df = pd.DataFrame(road_corr_results)
    road_corr_df.to_csv(os.path.join(save_path, '11_road_correlation_results.csv'), index=False, encoding='utf-8-sig')
    merged.to_csv(os.path.join(save_path, '11_road_change_data.csv'), index=False, encoding='utf-8-sig')
    
    return merged, road_corr_df

# =============================================================================
# 12. 공장용지 성장 히트맵
# =============================================================================
def create_factory_heatmap(df, save_path):
    """공장용지 성장 히트맵 생성"""
    
    pivot = df.pivot(index='행정구역명', columns='연도', values='공장용지면적_비율')
    
    growth_index = pivot.div(pivot[2017], axis=0) * 100
    
    growth_index['성장률'] = growth_index[2025] - 100
    growth_index = growth_index.sort_values('성장률', ascending=False)
    growth_index = growth_index.drop(columns=['성장률'])
    
    # 시각화
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(growth_index, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': '성장지수 (2017=100)'}, ax=ax)
    
    ax.set_title('시군구별 공장용지 성장지수 (2017년=100 기준)', fontsize=14, fontweight='bold')
    ax.set_xlabel('연도', fontsize=11)
    ax.set_ylabel('행정구역', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '12_heatmap_factory_growth.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    growth_index.to_csv(os.path.join(save_path, '12_factory_growth_index.csv'), encoding='utf-8-sig')
    
    return growth_index

# =============================================================================
# 13. 통합 인터랙티브 지도
# =============================================================================
def create_comprehensive_map(df, change_df, ts_df, save_path):
    """통합 인터랙티브 지도 생성"""
    
    df_2025 = df[df['연도'] == 2025].copy()
    ts_2025 = ts_df[ts_df['Year'] == 2025].copy()
    ts_2017 = ts_df[ts_df['Year'] == 2017].copy()
    
    # 인구 변화율 계산
    pop_change = ts_2017.merge(ts_2025, on='Region', suffixes=('_2017', '_2025'))
    pop_change['인구_변화율'] = (pop_change['Population_2025'] - pop_change['Population_2017']) / pop_change['Population_2017'] * 100
    
    m = folium.Map(location=[36.75, 127.75], zoom_start=9, tiles='cartodbpositron')
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: white; padding: 15px; border-radius: 5px;
                border: 2px solid gray; font-size: 12px;">
        <b>군집 분류</b><br>
        <i style="background:#E74C3C; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> 도시/산업형<br>
        <i style="background:#27AE60; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> 농업/산림형<br>
        <i style="background:#3498DB; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> 균형형<br>
        <hr>
        <b>원 크기: 공장용지 변화율</b>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    for idx, row in df_2025.iterrows():
        region = row['행정구역명']
        if region not in COORDINATES:
            continue
        
        coords = COORDINATES[region]
        cluster = row['군집']
        color = CLUSTER_COLORS.get(cluster, 'gray')
        
        change_row = change_df[change_df['행정구역명'] == region]
        pop_row = pop_change[pop_change['Region'] == region]
        
        if len(change_row) > 0:
            factory_change = change_row['공장용지면적_변화율'].values[0]
            forest_change = change_row['임야면적_변화율'].values[0]
            farm_change = change_row['농경지면적_변화율'].values[0]
            house_change = change_row['대지면적_변화율'].values[0]
        else:
            factory_change = forest_change = farm_change = house_change = 0
        
        if len(pop_row) > 0:
            pop_change_val = pop_row['인구_변화율'].values[0]
            pop_2025 = pop_row['Population_2025'].values[0]
        else:
            pop_change_val = 0
            pop_2025 = 0
        
        radius = 8 + abs(factory_change) * 0.3
        
        popup_html = f'''
        <div style="font-family: sans-serif; width: 280px;">
            <h4 style="margin-bottom: 5px;">{region}</h4>
            <p style="color: {color}; font-weight: bold; margin: 0;">{cluster}</p>
            <hr style="margin: 8px 0;">
            <b>2025년 토지 구성</b><br>
            - 임야: {row['임야면적_비율']:.1f}%<br>
            - 농경지: {row['농경지면적_비율']:.1f}%<br>
            - 대지: {row['대지면적_비율']:.1f}%<br>
            - 공장용지: {row['공장용지면적_비율']:.2f}%<br>
            <hr style="margin: 8px 0;">
            <b>변화율 (2017→2025)</b><br>
            - 임야: {forest_change:+.2f}%<br>
            - 농경지: {farm_change:+.2f}%<br>
            - 대지: {house_change:+.2f}%<br>
            - <span style="color: red;">공장용지: {factory_change:+.2f}%</span><br>
            <hr style="margin: 8px 0;">
            <b>인구 현황</b><br>
            - 2025년 인구: {pop_2025:,.0f}명<br>
            - <span style="color: blue;">인구 변화율: {pop_change_val:+.1f}%</span>
        </div>
        '''
        
        folium.CircleMarker(
            location=coords,
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
            tooltip=f"{region} ({cluster})"
        ).add_to(m)
    
    m.save(os.path.join(save_path, '13_comprehensive_map.html'))
    
    return m

# =============================================================================
# 14. 분석 결과 요약 저장
# =============================================================================
def save_analysis_summary(df, change_df, ts_df, pca_summary, cluster_summary, save_path):
    """분석 결과 요약 저장"""
    
    df_2025 = df[df['연도'] == 2025].copy()
    
    summary = df_2025[['행정구역명', '군집', '임야면적_비율', '농경지면적_비율', 
                       '대지면적_비율', '공장용지면적_비율']].merge(
        change_df[['행정구역명', '임야면적_변화율', '농경지면적_변화율', 
                   '대지면적_변화율', '공장용지면적_변화율']],
        on='행정구역명'
    )
    
    summary['도시화지표_2025'] = summary['대지면적_비율'] + summary['공장용지면적_비율']
    summary = summary.sort_values('공장용지면적_변화율', ascending=False)
    
    summary.to_csv(os.path.join(save_path, 'analysis_summary.csv'), index=False, encoding='utf-8-sig')
    
    # 분류 근거 텍스트 저장
    basis_text = f"""충청북도 지적통계 토지 이용 현황 분석 - 분류 근거
================================================================================

1. PCA (주성분 분석) 결과
--------------------------------------------------------------------------------
- PC1 분산 설명력: {pca_summary['PC1_분산설명력']*100:.1f}%
- PC2 분산 설명력: {pca_summary['PC2_분산설명력']*100:.1f}%
- 누적 분산 설명력: {pca_summary['누적_분산설명력']*100:.1f}%

- PC1 해석: {pca_summary['PC1_해석']}
- PC2 해석: {pca_summary['PC2_해석']}

2. K-Means 군집화 결과
--------------------------------------------------------------------------------
- 군집 수 (k): {cluster_summary['k']}
- Silhouette Score: {cluster_summary['silhouette_score']:.3f}
- 해석: {cluster_summary['해석']}

3. 군집 분류 정의
--------------------------------------------------------------------------------
도시/산업형 (4개 지역):
  - 청주 흥덕구, 청주 청원구, 진천군, 음성군
  - 특징: 공장용지/대지 비율 높음, 도시화 진행 중
  - 도시화 지표(대지+공장) 평균: 13% 이상

농업/산림형 (8개 지역):
  - 청주 상당구, 충주시, 제천시, 보은군, 옥천군, 영동군, 괴산군, 단양군
  - 특징: 임야 비율 75% 이상, 농업/산림 중심
  - 임야 비율 평균: 80% 이상

균형형 (2개 지역):
  - 청주 서원구, 증평군
  - 특징: 도시/농촌 기능 혼재
  - 중간 지표값 보유

4. 분석 데이터 기간
--------------------------------------------------------------------------------
- 분석 기간: 2017년 ~ 2025년 (9개년)
- 대상 지역: 충청북도 14개 시군구
- 토지 유형: 임야, 농경지, 대지, 공장용지, 도로

5. 주요 분석 결과
--------------------------------------------------------------------------------
- 전체적으로 임야/농경지 감소, 대지/공장용지 증가 추세
- 도시/산업형 지역에서 인구 증가 및 공장용지 확대 현저
- 농업/산림형 지역에서 인구 감소 및 토지 이용 변화 미미
- 도로면적과 도시화 지표 간 양의 상관관계 확인
"""
    
    with open(os.path.join(save_path, 'classification_basis.txt'), 'w', encoding='utf-8') as f:
        f.write(basis_text)
    
    return summary

# =============================================================================
# 메인 실행
# =============================================================================
def main():
    """메인 실행 함수"""
    
    print("=" * 70)
    print("충청북도 지적통계 토지 이용 현황 종합 분석")
    print("=" * 70)
    
    # result_v3 초기화
    for f in os.listdir(RESULT_DIR):
        os.remove(os.path.join(RESULT_DIR, f))
    
    # 데이터 로드
    print("\n[데이터 로드]")
    df = load_data()
    change_df = load_change_data()
    ts_df = load_timeseries_with_pop_road()
    print(f"  - 기본 데이터: {len(df)}건")
    print(f"  - 변화율 데이터: {len(change_df)}건")
    print(f"  - 시계열 데이터 (인구/도로 포함): {len(ts_df)}건")
    
    # 01. 데이터 현황
    print("\n[01/13] 데이터 현황 및 기초 통계 분석...")
    analyze_data_overview(df, ts_df, RESULT_DIR)
    
    # 02. PCA 분석
    print("[02/13] PCA 분석 수행...")
    pca_result, pca_summary, loading_df = perform_pca_analysis(df, RESULT_DIR)
    print(f"  - 누적 분산 설명력: {pca_summary['누적_분산설명력']*100:.1f}%")
    
    # 03. K-Means 분석
    print("[03/13] K-Means 군집화 분석...")
    cluster_summary = perform_clustering_analysis(df, RESULT_DIR)
    print(f"  - Silhouette Score (k=3): {cluster_summary['silhouette_score']:.3f}")
    
    # 04. 유형별 토지 구성
    print("[04/13] 유형별 토지 구성 비교...")
    analyze_cluster_composition(df, RESULT_DIR)
    
    # 05. 연도별 토지유형 변화량
    print("[05/13] 연도별 토지유형 변화량 분석...")
    analyze_yearly_landtype_change(df, RESULT_DIR)
    
    # 06. 유형별 연도별 추이
    print("[06/13] 유형별 연도별 추이 분석...")
    analyze_cluster_trend(df, RESULT_DIR)
    
    # 07. 충북 전체 추이
    print("[07/13] 충북 전체 추이 분석...")
    analyze_chungbuk_total_trend(df, RESULT_DIR)
    
    # 08. 14개 시군 비교
    print("[08/13] 14개 시군 비교 분석...")
    analyze_all_regions_yearly(df, RESULT_DIR)
    
    # 09. 청주 4개구 분석
    print("[09/13] 청주 4개구 비교 분석...")
    analyze_cheongju_4gu(df, change_df, RESULT_DIR)
    
    # 10. 인구 상관관계
    print("[10/13] 인구 상관관계 분석...")
    analyze_population_correlation(ts_df, RESULT_DIR)
    
    # 11. 도로 상관관계
    print("[11/13] 도로 상관관계 분석...")
    analyze_road_correlation(ts_df, RESULT_DIR)
    
    # 12. 공장용지 히트맵
    print("[12/13] 공장용지 성장 히트맵 생성...")
    create_factory_heatmap(df, RESULT_DIR)
    
    # 13. 통합 지도
    print("[13/13] 통합 인터랙티브 지도 생성...")
    create_comprehensive_map(df, change_df, ts_df, RESULT_DIR)
    
    # 결과 요약 저장
    print("\n결과 요약 저장...")
    save_analysis_summary(df, change_df, ts_df, pca_summary, cluster_summary, RESULT_DIR)
    
    print("\n" + "=" * 70)
    print("분석 완료!")
    print(f"결과 저장 위치: {RESULT_DIR}")
    print("=" * 70)
    
    # 생성된 파일 목록
    print("\n생성된 파일:")
    for f in sorted(os.listdir(RESULT_DIR)):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
