
Redes Neurais Artificiais (ANNs), ou simplesmente **redes neurais**, são modelos computacionais inspirados na estrutura e função das redes neurais biológicas. Consistem em nós interconectados (neurônios) que processam informações de maneira similar à forma como os neurônios no cérebro humano operam. ANNs são capazes de aprender a partir de dados, tornando-as ferramentas poderosas para várias tarefas, como reconhecimento de imagem, processamento de linguagem natural e tomada de decisão.

Redes neurais são a espinha dorsal de muitas aplicações modernas de IA, permitindo que máquinas aprendam com a experiência e melhorem seu desempenho ao longo do tempo. São particularmente eficazes em lidar com padrões complexos e grandes conjuntos de dados, tornando-as adequadas para uma ampla variedade de aplicações, desde visão computacional até reconhecimento de fala.

## Marcos Históricos

[timeline left alternate(./docs/2026.2/classes/ann/timeline.json)]

[^1]: **Modelo de Hodgkin–Huxley.**
Alan Hodgkin e Andrew Huxley desenvolvem um modelo matemático do potencial de ação em neurônios, descrevendo como os neurônios transmitem sinais através de impulsos elétricos. Este modelo é fundamental para compreender a dinâmica neural e influencia o desenvolvimento de redes neurais artificiais.
*Hodgkin, A. L., Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve.*
[:octicons-book-24:](https://doi.org/10.1113/jphysiol.1952.sp004764){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model){target='_blank'}
[:octicons-download-24:](https://www.its.caltech.edu/~jkenny/nb250c/papers/hodgkin_52_5.pdf){target='_blank'} [:medal:](https://www.nobelprize.org/prizes/medicine/1963/summary/){target='_blank'}.

[^2]: **Córtex Visual e Privação Monocular.**
David H. Hubel e Torsten N. Wiesel conduzem pesquisa pioneira sobre o córtex visual de gatos, demonstrando como a experiência visual molda o desenvolvimento neural. Seu trabalho sobre privação monocular mostra que privar um olho de entrada visual durante um período crítico leva a mudanças permanentes no córtex visual, destacando a importância da experiência na plasticidade neural.
*Hubel, D. H., & Wiesel, T. N. (1963). Effects of monocular deprivation in kittens.*
[:octicons-book-24:](https://doi.org/10.1007/bf00348878){target='_blank'}
[:octicons-download-24:](https://cw.fel.cvut.cz/b241/_media/courses/a6m33ksy/hubel-wiesel-1964-kittens.pdf){target='_blank'}
[:simple-youtube:](https://www.youtube.com/watch?v=KE952yueVLA&pp=0gcJCfwAo7VqN5tD){target='_blank'}
[:medal:](https://www.nobelprize.org/prizes/medicine/1981/summary/){target='_blank'}.

[^3]: **Neocognitron.** Kunihiko Fukushima desenvolve o Neocognitron, um dos primeiros modelos de rede neural convolucional (CNN) que imita a estrutura hierárquica do córtex visual. Este modelo é precursor das CNNs modernas e demonstra o potencial da extração hierárquica de features em tarefas de reconhecimento de imagem.
*Fukushima, K. (1980). Neocognitron: A new algorithm for pattern recognition tolerant of deformations and shifts in position.*
[:octicons-book-24:](https://doi.org/10.1007/BF00344251){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Neocognitron){target='_blank'}
[:octicons-download-24:](https://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf){target='_blank'}.

[^4]: **Redes de Hopfield.**
John Hopfield introduz as redes de Hopfield, um tipo de rede neural recorrente que pode servir como sistemas de memória associativa. Essas redes são capazes de armazenar e recuperar padrões, lançando as bases para desenvolvimentos posteriores em arquiteturas de redes neurais.
*Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities.*
[:octicons-book-24:](https://doi.org/10.1073/pnas.79.8.2554){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Hopfield_network){target='_blank'}
[:octicons-download-24:](https://www.dna.caltech.edu/courses/cs191/paperscs191/Hopfield82.pdf){target='_blank'} [:medal:](https://www.nobelprize.org/prizes/physics/2024/summary/){target='_blank'}.

[^5]: **Mapas Auto-Organizáveis (SOM).**
Teuvo Kohonen desenvolve os Mapas Auto-Organizáveis, um tipo de algoritmo de aprendizado não supervisionado que mapeia dados de alta dimensionalidade para uma grade de menor dimensão. SOMs são usados para clusterização e visualização de dados complexos, fornecendo insights sobre a estrutura dos dados.
*Kohonen, T. (1982). Self-organized formation of topologically correct feature maps.*
[:octicons-book-24:](https://doi.org/10.1007/BF00337288){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Self-organizing_map){target='_blank'}
[:octicons-download-24:](https://tcosmo.github.io/assets/soms/doc/kohonen1982.pdf){target='_blank'}. 

[^6]: **Redes Long Short-Term Memory (LSTM).**
Sepp Hochreiter e Jürgen Schmidhuber introduzem redes LSTM, um tipo de rede neural recorrente projetada para aprender dependências de longo prazo em dados sequenciais. Esta arquitetura aborda o problema do gradiente desvanecente em RNNs, possibilitando modelagem eficaz de dependências de longo prazo em dados sequenciais.
*Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.*
[:octicons-book-24:](https://doi.org/10.1162/neco.1997.9.8.1735){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Long_short-term_memory){target='_blank'}
[:octicons-download-24:](https://www.bioinf.jku.at/publications/older/2604.pdf){target='_blank'}".

[^7]: **Redes Residuais (ResNets).**
Kaiming He, Xiangyu Zhang, Shaoqing Ren e Jian Sun introduzem as Redes Residuais (ResNets), uma arquitetura de aprendizado profundo que usa conexões de salto para permitir que gradientes fluam mais facilmente por redes profundas. Esta arquitetura permite o treinamento de redes neurais muito profundas, melhorando significativamente o desempenho em tarefas de reconhecimento de imagem.
*He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition.*
[:octicons-book-24:](https://doi.org/10.1109/CVPR.2016.90){target='_blank'}
[:material-wikipedia:](https://en.wikipedia.org/wiki/Residual_network){target='_blank'}
[:octicons-download-24:](https://arxiv.org/pdf/1512.03385){target='_blank'}


---

--8<-- "docs/2026.2/classes/ann/quiz.pt.md"
