# Geodesic Dome Projection and Field-of-View Clipping  
## 지오데식 돔의 카메라 기반 투영 및 FOV 클리핑 수학적 정리 (Markdown 버전)

---

## 1. 서론

지오데식 돔은 정이십면체(icosahedron)를 기반으로 한 구면 분할 구조로,  
카메라 시점 투영, 시야각(FOV) 클리핑, 그리고 평면 투영 과정을 거쳐 표현될 수 있다.

본 문서에서는 다음을 정의한다:

- 정이십면체 기반 지오데식 구조 생성  
- 카메라 방향(azimuth, elevation, roll)에 따른 카메라 좌표계 구성  
- 3D → 카메라 공간 변환  
- FOV 기반 가시성 판정  
- FOV 평면 사각형 영역과의 클리핑  
- 2D 이미지 평면으로 최종 투영  

또한 **3D와 2D에서 보이는 삼각형 집합이 1:1로 대응하도록 조건을 통일한 모델**을 제시한다.

---

## 2. 정이십면체 및 지오데식 분할

### 2.1 정이십면체 정점 생성

황금비:

$$
\phi = \frac{1+\sqrt{5}}{2}
$$

정점 정규화:

$$
\hat{v}_i = \frac{v_i}{\|v_i\|}
$$

---

### 2.2 면 분할 (Frequency = \(f\))

한 면 \((v_0, v_1, v_2)\)에 대해 barycentric 분할:

$$
p(a,b,c)=\frac{a v_0 + b v_1 + c v_2}{a+b+c}, \qquad a+b+c=f
$$

구면 정규화:

$$
\hat{p}(a,b,c)=\frac{p(a,b,c)}{\|p(a,b,c)\|}
$$

면 전체 삼각형 수:

$$
N = 20 f^2
$$

---

## 3. 카메라 모델

카메라는 원점에 위치하며 방향(α, β, γ)을 가진다.

### 3.1 Forward 벡터

$$
\mathbf{f} =
(\cos\beta\cos\alpha,\;
 \cos\beta\sin\alpha,\;
 \sin\beta)
$$

### 3.2 Right / Up 벡터

$$
\mathbf{r}=
\frac{\mathbf{f} \times (0,0,1)}
{\|\mathbf{f} \times (0,0,1)\|}
$$

$$
\mathbf{u} = \mathbf{r} \times \mathbf{f}
$$

### 3.3 Roll 적용

$$
\begin{aligned}
\mathbf{r}' &= \cos\gamma\, \mathbf{r} + \sin\gamma\, \mathbf{u} \\
\mathbf{u}' &= -\sin\gamma\, \mathbf{r} + \cos\gamma\, \mathbf{u}
\end{aligned}
$$

카메라 행렬:

$$
R =
\begin{bmatrix}
(\mathbf{r}')^T \\
(\mathbf{u}')^T \\
\mathbf{f}^T
\end{bmatrix}
$$

---

## 4. 3D → 카메라 공간 변환

정점 
$$
\mathbf{v}_{cam}=R\mathbf{v}=
\begin{bmatrix}
x_{cam}\\ y_{cam}\\ z_{cam}
\end{bmatrix}
$$

---

## 5. FOV 기반 가시성 판정

각 정점 기준:

$$
\theta_x = \arctan\left(\frac{x_{cam}}{z_{cam}}\right)
$$

$$
\theta_y = \arctan\left(\frac{y_{cam}}{z_{cam}}\right)
$$

가시성 조건:

$$
|\theta_x| \le \frac{\mathrm{FOV}_x}{2}, \qquad
|\theta_y| \le \frac{\mathrm{FOV}_y}{2}, \qquad
z_{cam} > 0
$$

삼각형 T가 보이는 조건:

$$
\exists\, v \in T : v \text{ is visible}
$$

---

## 6. 투영 공식

2D 정규화 투영:

각 정점에 대해, 카메라 기준 수평/수직 각도(구면 좌표)를

$$
\theta_x = \arctan2(x_{cam}, z_{cam}), \qquad
\theta_y = \arctan2(y_{cam}, z_{cam})
$$

라고 할 때, 전체 FOV (풀 앵글) $\mathrm{FOV}_x, \mathrm{FOV}_y$ 를 기준으로  
정규화된 2D 좌표는 다음과 같이 정의한다:

$$
n_x = \frac{\theta_x}{\mathrm{FOV}_x/2}, \qquad
n_y = \frac{\theta_y}{\mathrm{FOV}_y/2}
$$

- 즉, $\theta_x = \pm \mathrm{FOV}_x/2$ 인 점들은 $n_x = \pm 1$ 로 매핑되고  
- $\theta_y = \pm \mathrm{FOV}_y/2$ 인 점들은 $n_y = \pm 1$ 로 매핑된다.

따라서 $[-1,1]\times[-1,1]$ 정규화 평면은  
“구면 FOV 콘(수평/수직 각도 $\pm \mathrm{FOV}/2$) 내부”에 정확히 대응한다.


---

## 7. FOV 사각형 클리핑

삼각형 투영 결과:

$$
T = \{(n_x^{(1)}, n_y^{(1)}),
      (n_x^{(2)}, n_y^{(2)}),
      (n_x^{(3)}, n_y^{(3)})\}
$$

FOV 정규화 사각형:

$$
C = [-1,1] \times [-1,1]
$$

클리핑 수행:

$$
T' = \mathrm{clip}(T, C)
$$

---

## 8. 3D ↔ 2D 완전 대응 정리

### 8.1 3D 가시 삼각형 집합

$$
\mathcal{V}_{3D} =
\left\{
T \mid
\exists v\in T :
z_{cam}>0,\;
|\theta_x| \le \tfrac{\mathrm{FOV}_x}{2},\;
|\theta_y| \le \tfrac{\mathrm{FOV}_y}{2}
\right\}
$$

### 8.2 2D 표시 집합

$$
\mathcal{V}_{2D} =
\left\{
\mathrm{clip}(\Pi(T), C)
\mid
T \in \mathcal{V}_{3D}
\right\}
$$

---

## ✔ 핵심 정리

$$
\boxed{
\mathcal{V}_{2D} \text{ 는 } \mathcal{V}_{3D} \text{ 와 정확히 동일한 삼각형 집합이다.}
}
$$

단,  
- FOV 바깥 영역만 잘려 나가며  
- 삼각형의 존재 여부는 항상 일치한다.

---

## 9. 결론

본 모델은 지오데식 돔의 카메라 투영 과정 전체를  
일관된 수학적 구조 하에서 정의하고,  
3D ↔ 2D 가시성 일치를 보장하는 견고한 기반을 제공한다.

---
