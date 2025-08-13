# Industrial AI Project

## 📌 개요 (Overview)
본 프로젝트는 **산업인공지능(Industrial AI)** 기술을 활용하여 실제 산업 현장의 문제를 해결하는 것을 목표로 합니다.  
품질 검사, 예지 정비, 전력 수요 예측 등 다양한 산업 데이터를 기반으로 **머신러닝·딥러닝 모델**을 설계, 구현, 검증합니다.

---

## 🎯 프로젝트 목표 (Objectives)
- **데이터 기반 의사결정** 지원
- **딥러닝/머신러닝 모델**을 통한 산업 문제 해결
- 산업 현장 맞춤형 **AI 솔루션** 개발 및 성능 최적화
- **실시간 분석 및 경보 시스템** 구현

---

## 🛠 사용 기술 (Tech Stack)
- **언어**: Python, C#
- **프레임워크**: TensorFlow, PyTorch, Scikit-learn
- **데이터 처리**: Pandas, NumPy
- **시각화**: Matplotlib, Seaborn, Plotly
- **DB/스토리지**: MySQL, Oracle
- **기타**: OpenCV (영상 처리), Flask/FastAPI (웹 서비스)

---

## 📂 데이터셋 (Datasets)
| 데이터명 | 설명 | 출처 |
|----------|------|------|
| UCI Household Electric Power Consumption | 전력 수요 예측 데이터 | UCI ML Repository |
| Custom Quality Inspection Images | 제조 품질 검사 이미지 | 자체 수집 |
| CCTV Emergency Detection Dataset | 화재/낙상 감지 영상 데이터 | 자체 라벨링 |

---

## 📈 주요 기능 (Key Features)
1. **품질 검사 불량 예측 모델**  
   - 제조 공정의 이미지 데이터를 기반으로 CNN 모델 학습  
   - 불량/양품 자동 분류
2. **전력 수요 예측 모델**  
   - 시계열 데이터를 LSTM, Transformer 기반으로 학습  
   - 단기/중기 예측 성능 비교
3. **실시간 CCTV 비전 분석**  
   - 화재, 낙상 등 위험 상황 감지  
   - 알람 및 SMS/Push 알림 연동

---

## 🧪 모델 성능 (Model Performance)
| 모델 | Task | Metric | Score |
|------|------|--------|-------|
| CNN | 품질검사 이미지 분류 | Accuracy | 94.2% |
| LSTM | 전력 수요 예측 | RMSE | 0.087 |
| Transformer | 전력 수요 예측 | RMSE | 0.072 |

---

## 📜 프로젝트 구조 (Project Structure)
