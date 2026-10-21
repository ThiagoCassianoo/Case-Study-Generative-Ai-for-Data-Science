[README (1).md](https://github.com/user-attachments/files/30992835/README.1.md)
# Case Study — Fine-Tuning + Quantização Leve (sobre o repo do curso Generative AI for Data Science)

Este diretório é um **case study construído em cima** do repositório de referência
[dgodevais/coursera-genai-for-data-science](https://github.com/dgodevais/coursera-genai-for-data-science)
(curso Microsoft/Coursera — certificado: [verificar aqui](https://coursera.org/verify/UKOGLFP89489)).

## O problema com rodar o original "como está"

Os notebooks `fine-tune-demo.ipynb` e `quantization-demo.ipynb` do repo original usam
**Mistral-7B-Instruct** com **QLoRA (4-bit) + `bitsandbytes` + `device_map="cuda:0"`**.
Isso **exige GPU NVIDIA com CUDA** — não roda em CPU, não roda de graça no GitHub Codespaces
padrão (sem GPU), e não roda no Google Colab gratuito sem fila de espera por GPU.

## A adaptação (documentada como decisão de engenharia, não como limitação escondida)

Mesma tarefa do original — classificar perguntas do Stack Overflow em `HQ` / `LQ_EDIT` / `LQ_CLOSE`
a partir de Título + Corpo + Tags — resolvida com as ferramentas certas pro orçamento certo:

| | Original | Este case |
|---|---|---|
| Modelo | Mistral-7B-Instruct (causal LM, 7B params) | `distilbert-base-uncased` (encoder, 66M params) |
| Por quê o encoder é melhor aqui | — | Classificação é tarefa nativa de encoder. Prompting um LLM causal pra classificar é mais caro e não é mais preciso pra esse tipo de problema. |
| Fine-tuning | QLoRA 4-bit (`peft`, `bitsandbytes`) | Fine-tuning completo em fp32 (`transformers.Trainer`) |
| Quantização | `BitsAndBytesConfig` 8-bit / 4-bit (exige CUDA) | `torch.quantization.quantize_dynamic` (nativo, CPU) |
| Custo de execução | GPU dedicada (paga) | Gratuito, roda em CPU no Codespaces |

Isso não é "a versão fácil". É a decisão de engenharia certa quando o objetivo é um classificador
de produção rodando sem GPU — e documentar esse raciocínio é o diferencial de portfólio: mostra
que a escolha de arquitetura foi pensada, não copiada do tutorial sem questionar.

## O que este case adiciona além do original

1. **Limpeza de dado real não documentada no repo original**: 1.1% das linhas de `train.csv`
   têm o rótulo (`Y`) nulo — tratado e registrado explicitamente em `src/data_prep.py`.
2. **Benchmark quantitativo real** de tamanho e latência (fp32 vs int8), não só uma demonstração conceitual.
3. **Clean Architecture**: preparação de dados isolada em `src/data_prep.py`, reutilizada pelos dois notebooks.

## Estrutura

```
./
├── README.md                              (este arquivo)
├── requirements.txt                       (deps leves, sem GPU)
├── src/
│   └── data_prep.py                        # limpeza + split 60/20/20, mesma proporção do original
├── notebooks/
│   ├── 01_finetune_lightweight.ipynb        # fine-tuning do DistilBERT
│   └── 02_quantize_and_benchmark.ipynb      # quantização dinâmica + benchmark fp32 vs int8
└── models/                                  # checkpoint gerado pelo notebook 01 (não versionado — ver .gitignore)
```

## Como rodar (dentro do Codespace do repositório)

```bash
# na raiz deste repositório
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# dataset (mesmo train.csv do repo original, ~32 MB, não versionado aqui)
curl -L -o train.csv https://raw.githubusercontent.com/dgodevais/coursera-genai-for-data-science/main/train.csv
```

Depois abra `notebooks/01_finetune_lightweight.ipynb` e rode célula a célula
(precisa de internet aberta pra baixar o checkpoint `distilbert-base-uncased` do Hugging Face —
funciona normalmente no Codespaces). Ao final, rode `02_quantize_and_benchmark.ipynb`,
que carrega o modelo salvo pelo notebook 1.

> **Nota de transparência:** os notebooks foram escritos e validados sinteticamente célula a célula
> (sintaxe, lógica de treino, mecânica de quantização testada com um modelo stand-in), mas o
> download do checkpoint real do Hugging Face precisa rodar num ambiente com internet irrestrita —
> não foi possível validar essa etapa específica no sandbox de geração deste projeto.
