# CDadosSeg #
 Resolução atividades ci1030-ERE2-CiênciaDeDados (source completo no repositório privado cds-ufpr)

## Índice :floppy_disk: ##
- [Para que serve este repositório?](#Para-que-serve-este-repositório?)
- [Atividades](#Atividades)
    - [T1](#T1)
    - [T2](#T2)
- [Projeto da disciplina](#Projeto-da-disciplina)
    - [T3](#T3)
    - [Atributos e representação](#Atributos-e-representação)
    - [Próximas etapas](#Próximas-etapas)
- [Referências](#Referências)

## Para que serve este repositório? ##
  Apresentação das atividades da disciplina de ciência de dados para segurança.

## Atividades ##
  Cada folder irá conter uma atividade, que será descrita nas seções seguintes.

### T1 ###

### T2 ###
  A primeira [Parte1](T2/Parte1/), consiste de um código `main.py` que deve receber como entrada um folder contendo `AndroidManifest.xml`. O folder com os respectivos manifests já esta presente [aqui](T2/Parte1/manifest).

  <details>
    <summary>Requisitos: (Clique para expandir!)</summary>

    * Python >= 3.8
    * pip3 com os seguintes pacotes:
      * xmltodict
      * argparse
  </details>

  Para executar a aplicação:
  ```
  python3 main.py --folder-name manifest/
  ```
  A segunda [Parte2](T2/Parte2), consiste de dois códigos `main.py` que deve receber um único arquivo executável como entrada e `main2.py` que deve receber dois arquivos executáveis.

  <details>
    <summary>Requisitos: (Clique para expandir!)</summary>

    * Python >= 3.8
    * pip3 com os seguintes pacotes:
      * pefile
      * argparse
  </details>

  Para executar a aplicação:
  ```
  python3 main.py --file-one Tibia_Setup.exe

  python3 main2.py --file-one Tibia_Setup.exe --file-two avira_pt-br_sptl1_89fae122b00d581b__pavwws.exe
  ```

## Projeto da disciplina ##

  Grupo: Tiago Heinrich (github: [h31nr1ch](https://github.com/h31nr1ch)) e Rodrigo Lemos (github: [Rodrigo-Lemos](https://github.com/Rodrigo-Lemos))

  O relatório pode ser encontrado [aqui](TrabalhoFinal/final.pdf). Onde é apresentado uma visão da problemática abordada e discussão dos resultados obtidos.

  <details>
    <summary>Requisitos: (Clique para expandir!)</summary>

    * Python >= 3.8
    * pip3 com os seguintes pacotes:
      * numpy
      * argparse
      * pandas
      * sklearn
      * pickle
      * matplotlib
  </details>

  Para executar a aplicação basta passar dois atributos de entrada que consistem na base do androzoo (especificamente o subset apresentado por [fabriciojoc](https://github.com/fabriciojoc)) e a base androDi. Ambas podem ser encontradas no diretório `dataset`.

  ```
  python3 main.py --dataset-androzoo dataset/datasetFabricio.csv --dataset-androDi dataset/azBalanceado.csv
  ```

### T3 ###

  A proposta a ser estudada, consiste na utilização do dataset Androzoo [1] em conjunto com um dataset gerado dinâmicamente (androDi) com objetivo de identificar malwares no Android. O dataset híbrido gerado dessa combinação tem como objetivo solucionar problemas existentes em ambos os datasets, como a ofuscação de código (problema de datasets estáticos) e a dificuldade na identificação de ataques online com o androDi devido a complexidade para coletar traços de execução das aplicações.

  Os dados do androDi consistem da distribuição de comandos executados no módulo do kernel Binder por cada uma das aplicações analisadas durante suas execuções no Android; em cada execução foram simulados 1000 iterações com o respectivo aplicativo. As aplicações foram escolhidas No dataset androzoo, de forma a obter os aplicativos rotulados e garantir que os aplicativos baixados não foram modificados após serem classificados; além disso, o androzoo já destaca um conjunto de características dos aplicativos analisados. Esta representação dinâmica é ideal para identificação de ataques em tempo real, uma vez que utiliza o comportamento do aplicativo em execução como característica para detecção, ao qual destaca uma das finalidades do modelo a ser proposto para realizar a identificação de ataques através de técnicas de machine learning.

  O androzoo, além de ser a origem da lista de aplicativos e seus rótulos, irá contribuir para o modelo de detecção através de características estáticas do dataset de forma a gerar um modelo de detecção híbrido. Um exemplo de característica estática a ser selecionada é a lista de permissões dos aplicativos; entretanto as demais features a compor o dataset ainda serão selecionadas.

  O dataset utilizado consiste em 201 aplicativos, rotulados em maliciosos ou não. Ele está distribuído de forma balanceada entre as duas classes existentes, sendo 96 aplicativos normais e 105 aplicativos maliciosos. Especificamente os dados encontrados no androDi são numéricos, aos quais as 37 features destacam uma representação de aparições em um traço. Uma amostra é apresentada a seguir:

  ```
  Apk,Label,BC_REPLY_SG,BC_TRANSACTION,BC_REPLY,BC_ACQUIRE_RESULT,BC_FREE_BUFFER,BC_INCREFS,BC_ACQUIRE,BC_RELEASE,BC_DECREFS,BC_INCREFS_DONE,BC_ACQUIRE_DONE,BC_ATTEMPT_ACQUIRE,BC_REGISTER_LOOPER,BC_ENTER_LOOPER,BC_EXIT_LOOPER,BC_REQUEST_DEATH_NOTIFICATION,BC_CLEAR_DEATH_NOTIFICATION,BC_DEAD_BINDER_DONE,BC_TRANSACTION_SG,BR_ERROR,BR_OK,BR_TRANSACTION,BR_ACQUIRE_RESULT,BR_DEAD_REPLY,BR_TRANSACTION_COMPLETE,BR_INCREFS,BR_ACQUIRE,BR_RELEASE,BR_DECREFS,BR_ATTEMPT_ACQUIRE,BR_NOOP,BR_SPAWN_LOOPER,BR_FINISHED,BR_DEAD_BINDER,BR_CLEAR_DEATH_NOTIFICATION_DONE,BR_FAILED_REPLY,BR_REPLY
  com.ForntYardIdeas.eshall,1,9876,35739,21682,0,82659,4845,4854,4333,4329,3683,3684,0,38,12,0,605,574,19,15458,0,0,51074,0,0,82741,3683,3682,3485,3492,0,0,37,0,19,574,0,31615
  ```

  O androDi foi explorado com o WEKA para a identificação de alguns comportamentos e melhor compreensão da distribuição do conjunto de dados. Permitindo identificar a importância e peso que features como BC_REPLY_SG, BC_REPLY e BR_TRANSACTION_COMPLETE tem sobre o dataset, também destacando atributos com baixa distribuição dentre as classes presentes.

  Na parte exploratória do AndroDi foi constatado alguns atributos que só possuem uma única classe, como BR_ACQUIRE_RESULT e BR_DEAD_REPLY. Estas classes não apresentam nenhuma contribuição ao conjunto de dados e serão removidas.

  ![alt text](https://github.com/h31nr1ch/CDadosSeg/blob/main/TrabalhoFinal/img/weka.png?raw=true)

  De forma a identificar características do androDi interessantes para o modelo foi realizado uma seleção de características utilizando o método RFECV nos dados normalizados por aplicativo, de forma a destacar a proporção entre os comandos em cada um dos aplicativos e desconsiderar o volume de chamadas dos mesmos. Como resultado foram selecionadas as características: ['BC_REPLY_SG', 'BC_TRANSACTION', 'BC_REPLY', 'BC_FREE_BUFFER', 'BC_INCREFS', 'BC_ACQUIRE', 'BC_DECREFS', 'BC_ACQUIRE_DONE', 'BC_REQUEST_DEATH_NOTIFICATION', 'BC_CLEAR_DEATH_NOTIFICATION', 'BC_TRANSACTION_SG', 'BR_TRANSACTION', 'BR_TRANSACTION_COMPLETE', 'BR_INCREFS', 'BR_ACQUIRE', 'BR_REPLY'].

### Atributos e representação ###

  O tipo de atributo presente no dataset, consiste de um conjunto numérico. As características foram extraídas ....

  Para o processo de normalização foi utilizando `MinMaxScaler()`.

### Próximas etapas ###

  Esta seção destaca as próximas etapas do trabalho.

  * Terminar a seleção de características do androzoo para a unificação dos datasets
  * Definição dos modelos e realização dos testes
  * Escrever os resultados obtidos

## Referências ##
[1] Allix, Kevin, et al. "Androzoo: Collecting millions of android apps for the research community." 2016 IEEE/ACM 13th Working Conference on Mining Software Repositories (MSR). IEEE, 2016.
