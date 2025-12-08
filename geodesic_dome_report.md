# Geodesic Dome Projection and Field-of-View Clipping

## 지오데식 돔의 카메라 기반 투영 및 FOV 클리핑에 관한 수학적 정리

------------------------------------------------------------------------

## 1. 서론

지오데식 돔은 정이십면체(icosahedron)를 기반으로 한 구면 분할 구조로,\
카메라 시점에서의 투영, FOV(Field of View) 클리핑, 그리고 평면화 과정을
거쳐\
시각적으로 표현할 수 있다.

본 보고서는 다음을 수학적으로 정의한다.

-   정이십면체 기반 지오데식 구조 생성
-   카메라 방향(azimuth, elevation, roll)의 벡터화를 통한 좌표계 구성
-   3D → 카메라 공간 변환
-   FOV 기반 가시성 판정
-   FOV 사각형과의 클리핑
-   2D 평면(이미지 평면)으로의 최종 투영

또한\
**왼쪽(3D 시각화)에서 보이는 삼각형과 오른쪽(2D 투영)에서 보이는
삼각형이 완전히 대응하도록**\
조건을 통일한 개선된 모델을 제시한다.

------------------------------------------------------------------------

## 2. 정이십면체 및 지오데식 분할

### 2.1 정이십면체 정점

정이십면체의 12개 정점은 황금비\
\[ arphi = rac{1+`\sqrt{5}`{=tex}}{2} \] 를 이용해 생성한다.

각 정점을 단위 벡터로 정규화하여 구에 내접시키면\
\[ `\hat{v}`{=tex}\_i = rac{v_i}{`\lVert `{=tex}v_i Vert} \]

------------------------------------------------------------------------

### 2.2 면 분할 (Frequency = f)

정이십면체 한 면 ((v_0, v_1, v_2))에 대해\
Barycentric 좌표로 f 등분한다.

\[ p(a,b,c)=rac{av_0 + bv_1 + cv_2}{a+b+c}, `\quad`{=tex} a+b+c = f \]

구면 정규화:

\[ `\hat{p}`{=tex}(a,b,c)=rac{p(a,b,c)}{`\lVert `{=tex}p(a,b,c)Vert} \]

각 면당 f²개의 삼각형, 전체는:

\[ N = 20 f\^2 \]

------------------------------------------------------------------------

## 3. 카메라 모델

카메라는 (0,0,0)에 위치하며 방향만 설정한다.

### 3.1 Forward 벡터

\[ `\mathbf{f}`{=tex} = (`\cos`{=tex}eta`\cos`{=tex}lpha,;
`\cos`{=tex}eta`\sin`{=tex}lpha,; `\sin`{=tex}eta) \]

### 3.2 Right, Up 벡터

\[ `\mathbf{r}`{=tex} = rac{`\mathbf{f}`{=tex} imes (0,0,1)}
{`\lVert`{=tex}`\mathbf{f}`{=tex} imes (0,0,1)Vert} \]

\[ `\mathbf{u}`{=tex} = `\mathbf{r}`{=tex} imes `\mathbf{f}`{=tex} \]

### 3.3 Roll 적용

\[ egin{aligned} `\mathbf{r}`{=tex}' &=
`\cos`{=tex}`\gamma`{=tex},`\mathbf{r}`{=tex} +
`\sin`{=tex}`\gamma`{=tex},`\mathbf{u}`{=tex} \\ `\mathbf{u}`{=tex}' &=
-`\sin`{=tex}`\gamma`{=tex},`\mathbf{r}`{=tex} +
`\cos`{=tex}`\gamma`{=tex},`\mathbf{u}`{=tex} \\end{aligned} \]

카메라 행렬:

\[ R = egin{bmatrix} (`\mathbf{r}`{=tex}')\^T \\
(`\mathbf{u}`{=tex}')\^T \\ (`\mathbf{f}`{=tex})\^T \\end{bmatrix} \]

------------------------------------------------------------------------

## 4. 3D → 카메라 좌표 변환

정점 (`\mathbf{v}`{=tex})에 대해:

\[ `\mathbf{v}`{=tex}*{cam} = R `\mathbf{v}`{=tex} = egin{bmatrix}
x*{cam} \\ y\_{cam} \\ z\_{cam} \\end{bmatrix} \]

------------------------------------------------------------------------

## 5. FOV 기반 가시성 판정 (3D)

각 정점에 대해:

\[ heta_x = rctan`\left`{=tex}(rac{x\_{cam}}{z\_{cam}}ight) \]

\[ heta_y = rctan`\left`{=tex}(rac{y\_{cam}}{z\_{cam}}ight) \]

정점이 FOV에 있으려면:

\[ \| heta_x\|
`\le `{=tex}rac{`\mathrm{FOV}`{=tex}\_x}{2},`\quad`{=tex} \| heta_y\|
`\le `{=tex}rac{`\mathrm{FOV}`{=tex}*y}{2},`\quad`{=tex} z*{cam} \> 0
\]

삼각형이 보이는 조건:

\[ `\exists `{=tex}v`\in `{=tex}T: v ext{ is in FOV} \]

이는 **왼쪽 3D에서 색칠되는 삼각형 집합**이다.

------------------------------------------------------------------------

## 6. 투영 (Projection)

정규화된 2D 좌표:

\[ n_x = rac{ an( heta_x)}{ an(`\mathrm{FOV}`{=tex}\_x/2)} \]

\[ n_y = rac{ an( heta_y)}{ an(`\mathrm{FOV}`{=tex}\_y/2)} \]

------------------------------------------------------------------------

## 7. FOV 사각형과의 클리핑

삼각형의 2D 투영:

\[ T = {(n_x^{(1)},n_y^{(1)}), (n_x^{(2)},n_y^{(2)}),
(n_x^{(3)},n_y^{(3)})} \]

FOV 정규화 사각형:

\[ C = \[-1,1\] imes \[-1,1\] \]

클리핑:

\[ T' = clip(T, C) \]

------------------------------------------------------------------------

## 8. 최종 규칙 (3D ↔ 2D 완전 대응)

### 8.1 3D 가시성 집합

\[ `\mathcal{V}`{=tex}*{3D} = `\left`{=tex}{ T `\mid `{=tex}
`\exists `{=tex}v`\in `{=tex}T: z*{cam}\>0,; \|
heta_x\|`\le`{=tex}rac{`\mathrm{FOV}`{=tex}\_x}{2},; \|
heta_y\|`\le`{=tex}rac{`\mathrm{FOV}`{=tex}\_y}{2} ight} \]

### 8.2 2D 표시 집합

\[ `\mathcal{V}`{=tex}*{2D} = `\left`{=tex}{ clip(`\Pi`{=tex}(T), C)
`\mid `{=tex}T`\in`{=tex}`\mathcal{V}`{=tex}*{3D} ight} \]

------------------------------------------------------------------------

## ✔ 핵심 정리

\[ oxed{ `\mathcal{V}`{=tex}*{2D} ext{ 는 } `\mathcal{V}`{=tex}*{3D}
ext{ 와 완전히 동일한 삼각형 집합이다.} } \]

단,\
- FOV를 벗어난 부분은 잘려 나가며\
- 삼각형 전체는 매핑 관계를 유지한다.

------------------------------------------------------------------------

## 9. 결론

본 보고서는 지오데식 돔을 카메라 시점에서 투영하는 과정을\
정확한 수학적 모델로 정리하고,\
3D와 2D 간 가시성 조건 일치를 보장하는 규칙을 명확히 기술하였다.

이 모델은 그래픽스, 시뮬레이션, 돔 매핑, 파노라마 시스템 등에 활용
가능하다.
