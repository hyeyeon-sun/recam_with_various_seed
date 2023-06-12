# 연구배경
 - 딥러닝 기술을 이용한 기술혁신이 증가하며 모델 훈련을 위해 더욱 많은 데이터가 필요
 - 특히 이미지 Segmentation의 경우 이미지 라벨 뿐 아니라 픽셀의 위치정보까지도 포함해야 학습할 수 있으므로 전처리 비용이 매우 큼
 - 약한 지도학습 & 반지도학습의 필요성 대두
 - 약한 지도학습 기반 이미지 segmentation의 성능을 증가시킴으로써 보다 적은 비용으로 segmentation이 가능하도록 함.

# 관련연구
## ScoreCAM
![image](https://github.com/hyeyeon-sun/recam_with_various_seed/assets/39080868/c75e6f12-15f3-47a5-8dd4-bcbbf6439e99)
- CAM의 가정과는 다르게 activation map의 weight와 target score는 비례하지 않는 경우가 많다는 점에서 착안한 연구
- Feature map을 업샘플링한 후 노멀라이징하여, 입력 이미지와 point-wise manipulation을 해준다. 
- 그 결과를 모델에 넣어서 나온 값과 원래의 베이스라인 이미지를 모델에 넣어서 나온 값을 빼서 Increase of Confidence를 구해준다.

## ReCAM
<img width="362" alt="image" src="https://github.com/hyeyeon-sun/recam_with_various_seed/assets/39080868/8433fcfa-ada0-4532-a444-e157916fdeb3">
- 다중 레이블 작업에서 각 클래스를 보다 정확하게 segmentation할 수 있도록 성능을 개선한 연구
- CAM의 경우 sum-over-class pooling 특성때문에 부정확한 segmentation 문제가 발생한다. 즉, Bus를 잘못 Segmentation 했다면 그 결과가 Person에도 영향을 미치는 것이다.
- 그런데, 다중 레이블 분류 작업의 경우 서로 다른 클래스간에 확률이 상호 의존적이지 않기 때문에, SCE를 추가 손실함수로 사용하여 각 클래스 별로 활성화 맵을 만들고 이를 통해 다시 CAM을 생성하는 방법론이다.

# 연구의 중요성
- ReCAM의 경우 다중레이블 분류작업에서 뛰어난 분류 성능을 보이며, 해당 연구에 CAM을 도입함으로써 Weakly-supervised Learning에서 Segmentation 작업에 있어 큰 성능 향상을 이룰 수 있다.
- 서로 다른 두가지 CAM 방식을 융합함으로써 보다 정확한 CAM 이미지를 추출할 수 있으며, 결과적으로 image segmentation 작업에 소요되는 비용을 크게 줄일 수 있다.

# 연구방법
## 코드 분석

<img width="803" alt="image" src="https://github.com/hyeyeon-sun/recam_with_various_seed/assets/39080868/f609876b-6f87-443f-aa28-fcfaaf25d63c">

## 코드 작성
<img width="715" alt="image" src="https://github.com/hyeyeon-sun/recam_with_various_seed/assets/39080868/a8d9a073-0d60-401f-85f1-69e0b95d0cfb">

## 데이터 투입하여 코드 수행

# 연구결과
<img width="713" alt="image" src="https://github.com/hyeyeon-sun/recam_with_various_seed/assets/39080868/5ea7148b-836e-49dc-9d23-fcd3b3d06940">

## CAM(Seed) 성능 비교
- Miou : 0.22 -> 0.29
 <img width="421" alt="image" src="https://github.com/hyeyeon-sun/recam_with_various_seed/assets/39080868/38694333-02ec-4b27-b648-2722cfaae7a3">


## Segmentation 성능 비교
- Miou : 0.28 -> 0.30
<img width="423" alt="image" src="https://github.com/hyeyeon-sun/recam_with_various_seed/assets/39080868/fb1d3d6e-876e-407e-83a7-8f64726cb8df">


# 참고 문헌
[1] Zhaozheng Chen, Tan Wang, Xiongwei Wu, Xian-Sheng Hua, Hanwang Zhang, Qianru
Sun. (2022)
Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation

[2] Jungbeom Lee, Eunji Kim, Sungmin Lee, Jangho Lee, Sungroh Yoon. (2019) FickleNet: Weakly and Semi-supervised Semantic Image Segmentation using Stochastic Inference

[3] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam. Devi Parikh, Dhruv Batra. (2019) Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

[4] Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang. (2020) Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
