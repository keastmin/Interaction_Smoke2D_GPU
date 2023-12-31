# Interaction_Smoke2D_GPU
Jos Stam의 논문 기반으로 작성된 Real Time Smoke Simulation을 CUDA를 이용하여 구현하였다. 그리고 사용자와의 상호작용을 추가하였다.  
그리드 기반의 연기 시뮬레이션으로 각 셀에 대하여 navier stokes 지배방정식을 red black gauss seidel 법을 통해 수치적으로 풀어냈다.  
기존의 OpenGL만 이용한 방식보다 그리드의 개수가 많아도 성능을 낼 수 있도록 GPU 프로그래밍을 적용하였다.  
구체를 숨겼을 때 보이는 색칠된 영역은 그리드와 구체의 충돌 감지 영역을 표현한 것이다.  
구체와 그리드에 대한 충돌 영역 체크는 구체의 중심 위치와 그리드의 각 셀의 중심 위치의 길이가 구체의 반지름보다 짧을 시 충돌 영역으로 간주한다.  
  


**사용 API**  
- OpenGL  
- GLEW  
- GLFW  
- GLM
- CUDA 12.2     
  

**시뮬레이션 환경**  
- CPU : intel i7-10 10700K  
- GPU : RTX 3070 BLACK EDITION OC D6 8GB  
- RAM : samsung DDR4-3200(8GB) x 2  
  

**참고 논문**  
- "Real-time fluid dynamics for games" by jos stam  
- "Stable fluids" by jos stam  
  

**컨트롤**  
- WASD : 카메라 이동
- 마우스 : 카메라 방향 전환, 장애물 구체 이동  
- 키보드 1번 : 연기 표시  
- 키보드 2번 : 속도 표시  
- Z : 소스항 추가 및 외력 추가  
- G : 장애물 구체 숨기기  
- C : 시뮬레이션 초기화  
- R : 시뮬레이션 전체 리셋  
  

**추가 사항**  
- main 코드의 add_force_source 함수를 통해 외력의 방향을 수정할 수 있음  
- density.cu에서 그리드의 충돌 영역에 대해 그리기 가능  
  
**결과**  
![Image Alt Text](https://github.com/keastmin/Interaction_Smoke2D_GPU/blob/main/image/smokeInteraction%20(1).jpg)  
![Image Alt Text](https://github.com/keastmin/Interaction_Smoke2D_GPU/blob/main/image/smokeInteraction%20(2).jpg)  
![Image Alt Text](https://github.com/keastmin/Interaction_Smoke2D_GPU/blob/main/image/smokeInteraction%20(3).jpg)  
