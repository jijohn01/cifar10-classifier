# cifar10-skipconnection
Deep Learning(CV) : cifar10 image classification practice
## requirement
- numpy (1.17.4)
- tensorflow ()
- matplotlib(3.1.2)

## 내용
### Skip connection이란? 
L 번째 레이어를 L+1번째 레이어만 연결하는 것이 아닌 L+n번째 레이어와도 연결하는 것을 말하며, 컴퓨터비전 분야에서 높은 성능을 보인 densenet에서 사용됩니다.
깊이가 많이 깊은 네트워크의 경우 input과 가까운 쪽의 레이어에게 학습의 힌트를 제공하여 빠른 학습과 성능개선에 도움을 주는 것으로 알려져 있습니다.
### 따라서 이 효과를 테스트하기 위해 간단한 문제에 적용하여 성능을 비교해 보았습니다.
## 결과
아래 결과와 같이 skip connection이 있는 경우 성능이 더 낮게 나타났습니다. 예상되는 이유는 문제가 비교적 쉽고, 모델의 깊이가 깊지 않아 skip connection의 효과를 얻기보다 오버피팅이 심해져 성능이 낮아진 것으로 보입니다.
<img width="958" alt="스크린샷 2020-05-16 오후 9 56 06" src="https://user-images.githubusercontent.com/28197373/82120311-0d3f7900-97c0-11ea-96ef-cd4b0a11e2c6.png">

