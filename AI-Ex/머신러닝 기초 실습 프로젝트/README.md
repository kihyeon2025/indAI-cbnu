AI-EX 머신러닝 기초 실습 프로젝트
Regression & PCA

1. 프로젝트 개요
본 프로젝트는 머신러닝의 핵심 기초 개념을 이해하기 위해
데이터 전처리부터 차원 축소, 회귀 모델 학습, 성능 평가,
그리고 과적합 방지 기법까지 단계별로 직접 구현하고 실습한
AI-EX 제출용 머신러닝 기초 프로젝트입니다.

수업 실습 코드를 기반으로 하되,
라이브러리에 의존하지 않고 알고리즘의 동작 원리와
수식적 의미를 코드로 확인하는 데 목적을 두었습니다.

2. 프로젝트 목적
- 머신러닝 기본 알고리즘의 이론 및 구현 이해
- 데이터 전처리와 학습 안정성의 중요성 파악
- 회귀 문제에서 전반적인 모델링 흐름 정리
- AI-EX 포트폴리오 제출용 실습 결과물 구성

3. 주요 학습 내용

3.1 데이터 정규화 (Normalization)
- Min-Max Scaling
- Standardization (Z-score)
- 정규화 적용 전후 학습 안정성 비교

파일:
lec03_3_normalization_for_student.ipynb

3.2 PCA (Principal Component Analysis)
- 공분산 행렬 기반 PCA 직접 구현
- 고유값 및 고유벡터 계산
- 주성분 선택 및 차원 축소
- sklearn PCA 결과와 비교

파일:
lec04_pca_v2_for_student.ipynb

3.3 Gradient Descent
- 비용 함수(Cost Function) 정의
- 학습률에 따른 수렴 특성 분석
- 반복 최적화 과정 구현

파일:
lec05_1_gradient_descent_for_student.ipynb

3.4 선형 회귀 (Linear Regression)

단변량 선형 회귀
- 단일 입력 변수 회귀
- Gradient Descent 기반 파라미터 학습

파일:
lec05_2_linear_regression_univariate_for_student.ipynb

다변량 선형 회귀
- 다수 입력 변수 회귀
- 행렬 연산 기반 학습
- 정규화 필요성 확인

파일:
lec05_3_linear_regression_multivariate_for_student.ipynb

3.5 모델 평가
- R² (결정계수)
- 회귀 성능 정량 평가 방법 이해

파일:
lec07_1_r2_for_student.ipynb

3.6 Regularization
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- 과적합 문제 분석
- Bias-Variance Tradeoff 이해

파일:
lec07_regularization_v2_for_student.ipynb

4. 사용 데이터셋
Boston Housing Dataset
- 주택 가격 예측을 위한 회귀 데이터셋
- Feature: 범죄율, 방 개수, 위치 정보 등
- Target: 주택 가격 (MEDV)

파일:
boston_house.pkl

5. 사용 기술
- Python
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

6. 프로젝트 특징 요약
- 머신러닝 전처리부터 학습, 평가까지 전 과정 실습
- PCA와 Gradient Descent를 수식 기반으로 직접 구현
- 회귀 모델 성능 평가 및 과적합 방지 기법 적용
- AI-EX 제출을 고려한 체계적인 실습 프로젝트 구성

