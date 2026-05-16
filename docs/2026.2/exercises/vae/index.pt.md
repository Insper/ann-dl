!!! success inline end "Prazo e Entrega"

    :date: 26.out (domingo)
    
    :clock1: Commits até 23:59

    :material-account: Individual

    :simple-github: Enviar o Link do GitHub Pages (sim, **apenas** o link das pages) via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

**Atividade: Implementação de VAE**

Neste exercício, você irá implementar e avaliar um Autoencoder Variacional (VAE) no dataset MNIST ou Fashion MNIST. O objetivo é entender a arquitetura, o processo de treinamento e o desempenho dos VAEs.


## Instruções

1. **Preparação dos Dados**:

    - Carregue o dataset MNIST/Fashion MNIST;
    - Normalize as imagens para o intervalo [0, 1];
    - Divida o dataset em conjuntos de treino e validação.

2. **Implementação do Modelo**:

    - Defina a arquitetura do VAE, incluindo as redes de encoder e decoder;
    - Implemente o truque de reparametrização.

3. **Treinamento**:

    - Treine o VAE no dataset MNIST/Fashion MNIST;
    - Monitore a perda e gere reconstruções durante o treinamento.

4. **Avaliação**:

    - Avalie o desempenho do VAE no conjunto de validação;
    - Gere novas amostras a partir do espaço latente aprendido.

5. **Visualização**:

    - Visualize imagens originais e reconstruídas;
    - Visualize o espaço latente (em caso de espaço latente até 3D, caso contrário use visualização reduzida, ex: t-SNE, UMAP ou PCA).

6. **Relatório**:

    - Resuma suas descobertas, incluindo desafios enfrentados e insights obtidos;
    - Inclua visualizações de reconstruções e espaço latente.

7. **Crédito Extra (Opcional)**:

    - Experimente o mesmo dataset com um Autoencoder (AE) e compare os resultados com o VAE;
    - Experimente com diferentes dimensões do espaço latente e reporte os efeitos na qualidade de reconstrução e geração de amostras.


!!! danger "Diretrizes Importantes"

    Esta é uma **atividade individual**. Você deve completar o trabalho por conta própria. Colaboração não é permitida, mas você pode discutir conceitos gerais com seus pares ou instrutores.
    
    Você pode usar o MLP do zero construído no exercício anterior, mas pode usar qualquer framework que preferir (ex: PyTorch, TensorFlow, Keras). Ferramentas de IA também podem ser usadas. ==MAS== lembre-se que o objetivo principal é entender a arquitetura e o processo de treinamento do VAE — **você deve ser capaz de explicar todas as partes do código e análise enviados**.

**Notas Importantes:**

- O entregável deve ser enviado em **GitHub Pages**. Existe um template do curso — [template](https://hsandmann.github.io/documentation.template/){target='_blank'};
- **O prazo não é estendido** — **NENHUMA EXCEÇÃO** para entregas atrasadas.
- **Colaboração com IA é permitida**, mas o aluno **DEVE ENTENDER** e explicar todo o código. **PROVAS ORAIS** podem ser realizadas.

**Critérios de Nota:**

| Critério | Descrição |
|:--------:|-------------|
| **3 pts** | Correção da implementação do VAE |
| **1 pt** | Treinamento e Avaliação: Procedimento de treinamento adequado, monitoramento da perda e avaliação no conjunto de validação. |
| **2 pts** | Amostragem: Qualidade das amostras geradas. |
| **2 pts** | Espaço Latente: Qualidade da representação do espaço latente aprendido. |
| **1 pt** | Visualizações: Qualidade e clareza dos plots. |
| **1 pt** | Qualidade do Relatório: Clareza, organização e completude do relatório. |
