# U-Net - Monocular images depth estimation

Repositório para implementação de projetos relacionados à dissertação do mestrado de computação aplicada.

## Proposta

Comparar desempenho de modelos baseados em U-Net (U-Net original) e Transformers (TransUnet, AttentionUnet) com outras implementações para a tarefa de estimativa de profundidade.

## Configurações para testar viabilidade do projeto

Modelo: U-Net (tradicional)

### Modo de leitura: [cv2.IMREAD_ANYDEPTH](https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#gga61d9b0126a3e57d9277ac48327799c80a0b486c93c25e8a0b0712681bb7254c18)

| done | width | height | filters (min) | filters (max) |
|:---:|:---:|:---:|:---:|:---:|
| <li>- [ ] </li> | 256 | 256 | 64 | 512 |
| <li>- [ ] </li> | 256 | 256 | 64 | 1024 |
| <li>- [ ] </li> | 512 | 512 | 64 | 512 |
| <li>- [ ] </li> | 512 | 512 | 64 | 1024 |

### Modo de leitura: [cv2.COLOR_BGR2GRA](https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a353a4b8db9040165db4dacb5bcefb6ea)

| done | width | height | filters (min) | filters (max) |
|:---:|:---:|:---:|:---:|:---:|
| <li>- [ ] </li> | 256 | 256 | 64 | 512 |
| <li>- [ ] </li> | 256 | 256 | 64 | 1024 |
| <li>- [ ] </li> | 512 | 512 | 64 | 512 |
| <li>- [ ] </li> | 512 | 512 | 64 | 1024 |

## Implementações

### Ambientes suportados
- [x] Google Colab
- [x] Kaggle
- [x] Local

### Features
- [ ] Fixar seed para reprodutibilidade dos resultados
- [x] Salvar modelo a cada época
- [x] Salvar resultados a cada época
- [x] Gerar CSV com resultados após finalizar caso de teste
- [x] Continua execução inativa após algum tempo
- [ ] Notebook para carregar modelo e exibir comparações na base de teste

### Métricas implementadas
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

