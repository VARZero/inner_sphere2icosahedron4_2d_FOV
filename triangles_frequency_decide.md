# 📘 지오데식 돔(Geodesic Dome)의 Frequency와 Pixel Resolution 관계  
## —— 해상도·FOV·삼각형 해상도의 일반화 수학 모델 ——

---

## 목차
1. [서론](#서론)  
2. [지오데식 돔 분할 모델](#지오데식-돔-분할-모델)  
3. [정이십면체의 기본 각도 구조](#정이십면체의-기본-각도-구조)  
4. [픽셀의 각 해상도](#픽셀의-각-해상도)  
5. [Edge 기반 Frequency 공식](#edge-기반-frequency-공식)  
6. [Area 기반 Frequency 공식](#area-기반-frequency-공식)  
7. [두 모델의 비교](#두-모델의-비교)  
8. [예시 계산: 1920×1080, FOV 90°×60°](#예시-계산-1920×1080-fov-90×60)  
9. [결론](#결론)  
10. [부록: 코드로 보는 요약](#부록-코드로-보는-요약)

---

# 서론

지오데식 돔을 시각화하거나 광학 시뮬레이션을 할 때,  
돔의 **frequency (분할 레벨)** 은 화면에서 삼각형이 얼마나 작고 촘촘하게 보일지를 결정하는 핵심 인자이다.

이 문서는 다음을 목표로 한다:

- 해상도와 FOV가 어떤 frequency를 요구하는지  
- 삼각형 하나의 크기를 픽셀 기준으로 제어하려면  
- 삼각형 개수와 화면 픽셀 개수를 맞추려면  
- 해상도가 바뀌어도 자동으로 f를 계산할 수 있도록  

이를 위해 **일반화된 두 개의 모델**을 제시한다:

1. **Edge 기반 모델** – 삼각형 한 변이 k 픽셀로 보이도록 하는 식  
2. **Area 기반 모델** – 반구의 삼각형 개수를 화면 픽셀 개수와 맞추는 식

---

# 지오데식 돔 분할 모델

정이십면체를 frequency \( f \) 로 분할하면:

- 한 면(삼각형)의 작은 삼각형 수  
  \[
  N_\text{face} = 2 f^2
  \]

- 전체 20면  
  \[
  N_\text{total} = 40 f^2
  \]

- 카메라에서 보이는 반구 영역  
  \[
  N_\text{visible} \approx 20 f^2
  \]

---

# 정이십면체의 기본 각도 구조

정이십면체의 한 변이 지구 중심에서 만드는 중심각:

\[
\theta_0 \approx 63.434948^\circ
\]

Frequency \( f \) 로 나누면 작은 삼각형 한 변의 각도는:

\[
\theta_\text{edge, small} \approx \frac{\theta_0}{f}
\]

---

# 픽셀의 각 해상도

화면 해상도를 \( W \times H \),  
FOV를 \( \text{FOV}_x \times \text{FOV}_y \) 라고 하면:

### 가로 픽셀 하나가 차지하는 각도
\[
\Delta\theta_{\text{px,x}} = \frac{\text{FOV}_x}{W}
\]

### 세로 픽셀 하나가 차지하는 각도
\[
\Delta\theta_{\text{px,y}} = \frac{\text{FOV}_y}{H}
\]

---

# Edge 기반 Frequency 공식

목표:
> 작은 삼각형 한 변이 화면에서 **k 픽셀** 정도로 보이도록 한다.

조건식:

\[
\theta_\text{edge, small}
\approx
k \cdot \Delta\theta_{\text{px,x}}
\]

정리하면:

\[
\frac{\theta_0}{f} \approx 
k \cdot \frac{\text{FOV}_x}{W}
\]

따라서 frequency \( f \):

---

## ⭐ 최종 Edge 기반 공식

\[
\boxed{
f \approx \frac{\theta_0}{k} \cdot \frac{W}{\text{FOV}_x}
}
\]

- \(W\): 가로 해상도  
- \(\text{FOV}_x\): 가로 시야각  
- \(k\): 삼각형 한 변을 몇 픽셀로 보이게 할 것인가  
- \(\theta_0 \approx 63.4^\circ\)

---

# Area 기반 Frequency 공식

반구가 화면에 원 형태로 투영된다고 가정하면  
그 원의 픽셀 반지름은:

\[
R_{\text{px}} \approx \frac{\min(W, H)}{2}
\]

원 내부 픽셀 수:

\[
N_\text{pixel(circle)} \approx \pi R_{\text{px}}^2
\]

반구 메시 삼각형 수:

\[
N_\text{visible} \approx 20 f^2
\]

샘플링 밀도를 맞춘다면:

\[
20 f^2 = \pi R_{\text{px}}^2
\]

따라서:

---

## ⭐ 최종 Area 기반 공식

\[
\boxed{
f \approx \sqrt{\frac{\pi}{20}}\, R_{\text{px}}
}
\]

보다 정밀하게는:

\[
R_{\text{px}} =
\frac{1}{2}
\min\left(
H,\,
W \cdot \frac{\text{FOV}_y}{\text{FOV}_x}
\right)
\]

---

# 두 모델의 비교

| 목적 | 추천 모델 | 설명 |
|------|----------|------|
| 삼각형 한 변을 k픽셀로 제어하고 싶다 | **Edge 기반** | 가장 직관적이고 제어 가능 |
| 삼각형 개수 ≈ 픽셀 개수로 맞추고 싶다 | **Area 기반** | 광학 샘플링 시 유용 |
| 시각적으로 부드러운 결과 원함 | Edge 기반(k=4~8) | 현실적으로 가장 실용적 |

---

# 예시 계산: 1920×1080, FOV 90°×60°

해상도:
- \( W = 1920 \)
- \( H = 1080 \)

FOV:
- \( \text{FOV}_x = 90^\circ \)
- \( \text{FOV}_y = 60^\circ \)

---

## Edge 기반 (k=4)

\[
f \approx 
\frac{63.4}{4} \cdot \frac{1920}{90}
\approx 338
\]

작은 삼각형 한 변 ≈ 4픽셀.

---

## Area 기반

\[
R_{\text{px}} \approx 540
\]

\[
f \approx \sqrt{\frac{\pi}{20}} \cdot 540
\approx 214
\]

반구의 삼각형 개수 ≈ 화면 픽셀 개수 수준.

---

# 결론

지오데식 돔의 frequency는 해상도·FOV·삼각형 크기에 따라 다음 공식을 통해 일반화된다.

---

## 🎯 최종 공식 요약

### Edge 기반 (삼각형 변 ≈ k픽셀)
\[
f \approx \frac{\theta_0}{k} \cdot \frac{W}{\text{FOV}_x}
\]

### Area 기반 (삼각형 개수 ≈ 픽셀 개수)
\[
f \approx \sqrt{\frac{\pi}{20}}\, R_{\text{px}}
\]

---

# 부록: 코드로 보는 요약

```python
import math

def freq_edge_based(W, FOVx_deg, k_pixels, theta0=63.434948):
    """Edge 기반 frequency 계산"""
    return theta0 * W / (k_pixels * FOVx_deg)

def freq_area_based(W, H):
    """Area 기반 frequency 계산"""
    Rpx = min(W, H) / 2
    return math.sqrt(math.pi / 20.0) * Rpx
