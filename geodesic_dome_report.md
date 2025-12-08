# Geodesic Dome Projection & FOV Clipping  
### 지오데식 돔의 카메라 기반 투영 및 FOV 클리핑 수학적 정리

---

## 📌 개요

지오데식 돔(geodesic dome)은 정이십면체 기반의 구면 분할 구조이며,  
카메라 투영 · FOV(Field of View) 클리핑 · 2D 평면 투영 과정이 필요하다.

이 문서는 다음을 정리한다:

- 정이십면체 기반 지오데식 돔 생성  
- 카메라 방향(azimuth, elevation, roll)으로부터의 좌표계 구성  
- 월드 좌표 → 카메라 좌표 변환  
- FOV 기반 가시성 판정  
- 2D 정규화 평면에서의 클리핑  
- **3D 가시 삼각형 집합과 2D 투영 삼각형 집합의 완전한 대응 관계 정리**  

---

## 🧱 1. 정이십면체 및 지오데식 분할

### 🔹 황금비

$$
\phi = \frac{1+\sqrt{5}}{2}
$$

### 🔹 정점 정규화

$$
\hat{v}_i = \frac{v_i}{\|v_i\|}
$$

---

### 🔹 면 분할 (Frequency = f)

Barycentric 분할:

$$
p(a,b,c) = \frac{a v_0 + b v_1 + c v_2}{a+b+c}, \qquad a+b+c=f
$$

구면 정규화:

$$
\hat{p}(a,b,c) = \frac{p(a,b,c)}{\|p(a,b,c)\|}
$$

전체 삼각형 수:

$$
N = 20 f^2
$$

---

## 🎥 2. 카메라 모델

카메라는 원점에 있고 방향만 설정한다(azimuth = $\alpha$, elevation = $\beta$, roll = $\gamma$).

### 🔹 Forward 벡터

$$
\mathbf{f} =
(\cos\beta\cos\alpha,\; \cos\beta\sin\alpha,\; \sin\beta)
$$

### 🔹 Right / Up 벡터

$$
\mathbf{r} =
\frac{\mathbf{f} \times (0,0,1)}
{\|\mathbf{f} \times (0,0,1)\|}
$$

$$
\mathbf{u} = \mathbf{r} \times \mathbf{f}
$$

### 🔹 Roll 적용

$$
\mathbf{r}' = \cos\gamma\,\mathbf{r} + \sin\gamma\,\mathbf{u}
$$

$$
\mathbf{u}' = -\sin\gamma\,\mathbf{r} + \cos\gamma\,\mathbf{u}
$$

카메라 회전행렬:

$$
R =
\begin{bmatrix}
(\mathbf{r}')^T \\
(\mathbf{u}')^T \\
\mathbf{f}^T
\end{bmatrix}
$$

---

## 🔄 3. 3D → 카메라 좌표 변환

정점 $\mathbf{v}$에 대한 카메라 좌표:

$$
\mathbf{v}_{cam} = R\,\mathbf{v}
= (x_{cam},\, y_{cam},\, z_{cam})^T
$$

---

## 👁️ 4. FOV 기반 가시성 판정

각 정점에 대해:

$$
\theta_x = \arctan\left(\frac{x_{cam}}{z_{cam}}\right),
\qquad
\theta_y = \arctan\left(\frac{y_{cam}}{z_{cam}}\right)
$$

가시 조건:

$$
|\theta_x| \le \frac{\mathrm{FOV}_x}{2}, \qquad
|\theta_y| \le \frac{\mathrm{FOV}_y}{2}, \qquad
z_{cam} > 0
$$

삼각형 $T$의 가시성:

$$
\exists\, v \in T : v \text{ is visible}
$$

---

## 🖥️ 5. 2D 정규화 투영

정규화된 2D 좌표:

$$
n_x = \frac{\tan(\theta_x)}{\tan(\mathrm{FOV}_x/2)},
\qquad
n_y = \frac{\tan(\theta_y)}{\tan(\mathrm{FOV}_y/2)}
$$

---

## ✂ 6. FOV 사각 클리핑

투영된 삼각형:

$$
T = \{(n_x^{(1)}, n_y^{(1)}),
      (n_x^{(2)}, n_y^{(2)}),
      (n_x^{(3)}, n_y^{(3)})\}
$$

FOV 정규화 공간:

$$
C = [-1,1] \times [-1,1]
$$

클리핑:

$$
T' = \mathrm{clip}(T, C)
$$

---

## 🔗 7. 3D ↔ 2D 삼각형 대응 관계

### 🔹 3D 가시 삼각형 집합

$$
\mathcal{V}_{3D} =
\lbrace T \mid
\exists v \in T,\;
z_{cam} > 0,\;
|\theta_x| \le \mathrm{FOV}_x/2,\;
|\theta_y| \le \mathrm{FOV}_y/2
\rbrace
$$

### 🔹 2D 표시 삼각형 집합

$$
\mathcal{V}_{2D} =
\lbrace \mathrm{clip}(\Pi(T), C) \mid T \in \mathcal{V}_{3D} \rbrace
$$

---

## 🧩 핵심 정리

$$
\boxed{
\mathcal{V}_{2D}
\text{ 는 }
\mathcal{V}_{3D}
\text{ 와 정확히 동일한 삼각형 집합이다.}
}
$$

- FOV 밖 영역만 잘려 나가며  
- 삼각형의 “존재 여부”는 3D/2D에서 완전히 일치한다.

---

## ✅ 결론

이 모델은 지오데식 돔을 카메라 시점에서 정확하게 투영하는 데 필요한  
**수학적·기하학적 규칙을 통합 정리**한다.  

그래픽스, 돔 프로젝션, 파노라마 렌더링 등 다양한 분야에서 적용 가능하다.
