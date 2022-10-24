# U-Net - Monocular images depth estimation

Repositório para implementação de projetos relacionados à dissertação do mestrado de computação aplicada.

## Checklist de implementações

Testar nos ambientes de execução:
- [x] Google Colab
- [x] Kaggle
- [ ] Local

Features:
- [x] Salvar modelo a cada época
- [x] Salvar resultados a cada época
- [x] Gerar CSV com resultados após finalizar caso de teste
- [x] Continua execução inativa após algum tempo


## Métricas implementadas
- [x] Threshold (δ¹, δ², δ³)
- [x] Abs. Relative Difference
- [x] Squared Relative Difference
- [x] RMSE linear
- [x] RMSE log
- [x] log 10


## Arquitetura

Para simplificar o entendimento do fluxo de execução da aplicação que gerencia a execução das etapas de treinamento, avaliação e teste, abaixo está os três primeiros níveis da modelagem C4.

### C4 - Nível 1
<p align="center">
  <img width="550px" src="https://user-images.githubusercontent.com/37306576/197453331-94ec15c9-d277-4aa3-880d-85279dc121f6.svg"/>
</p>

### C4 - Nível 2
<p align="center">
  <img width="min(600px, 50%)" src="https://user-images.githubusercontent.com/37306576/197453334-0bbee1ae-492d-46f1-be8b-79b71e93439a.svg"/>
</p>

### C4 - Nível 3
<p align="center">
  <img src="https://user-images.githubusercontent.com/37306576/197453332-62a6becb-17c3-4eb0-9400-6688d6584edf.svg"/>
</p>

