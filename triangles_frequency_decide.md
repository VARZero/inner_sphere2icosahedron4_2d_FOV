# 📘 Area 기반 Geodesic Frequency 공식

## 개요

Area 기반 모델은 다음 목표를 가진다:

- **화면에 투영된 돔의 픽셀 면적**과  
- **지오데식 돔 반구의 작은 삼각형 개수**

를 서로 대응시켜,  
픽셀당 메시 밀도(sample density)가 일정하도록 frequency f 를 계산하는 공식이다.

---

# 1. 화면에서 돔이 차지하는 원형 영역

돔이 화면에 투영되면 원형 형태로 보인다고 가정한다.  
이 원의 픽셀 반지름은 다음과 같이 근사할 수 있다.

$$
R_{\text{px}} \approx \frac{\min(W, H)}{2}
$$

보다 정밀한 근사:

$$
R_{\text{px}} =
\frac{1}{2}
\min\left(
H,\;
W \cdot \frac{\text{FOV}_y}{\text{FOV}_x}
\right)
$$

- W : 가로 해상도  
- H : 세로 해상도  
- FOV_x : 가로 FOV(도 단위)  
- FOV_y : 세로 FOV(도 단위)

---

# 2. 화면 원형 영역의 픽셀 수

$$
N_\text{pix(circle)} \approx \pi R_{\text{px}}^2
$$

---

# 3. 지오데식 돔 반구의 삼각형 개수

정이십면체 기반 frequency f 지오데식 돔의 반구 삼각형 수는:

$$
N_\text{visible} \approx 20 f^2
$$

---

# 4. 픽셀 밀도 = 메시 밀도 조건

두 값을 동일하게 맞춘다:

$$
20 f^2 = \pi R_{\text{px}}^2
$$

이 조건은 “삼각형 샘플링 밀도와 픽셀 밀도를 일치시키는” 상황을 의미한다.

---

# ⭐ 5. 최종 Area 기반 Frequency 공식

$$
f \approx \sqrt{\frac{\pi}{20}} \, R_{\text{px}}
$$

단일 공식으로 정리하면:

$$
f \approx \sqrt{\frac{\pi}{20}}
\cdot
\frac{\min(W, H)}{2}
$$

---

# 6. 파이썬 구현

```python
import math

def freq_area_based(W, H):
    """Area 기반 frequency 계산 (공식 B)"""
    Rpx = min(W, H) / 2
    return math.sqrt(math.pi / 20.0) * Rpx
