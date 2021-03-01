# CDadosSeg #
 Resolução atividades ci1030-ERE2-CiênciaDeDados (source completo no repositório privado cds-ufpr)

## Índice :floppy_disk: ##
- [Para que serve este repositório?](#Para-que-serve-este-repositório?)
- [Atividades](#Atividades)
    - [T2](#T2)
    - [T3](#T3)
    - [T4](#T4)
    - [Projeto da disciplina](#Projeto-da-disciplina)

## Para que serve este repositório? ##
  Apresentação das atividades da disciplina de ciência de dados para segurança.

## Atividades ##
  Cada folder irá conter uma atividade, que será descrita nas seções seguintes.

### T2 ###
  A primeira [Parte1](T2/Parte1/), consiste de um código `main.py` que deve receber como entrada um folder contendo `AndroidManifest.xml`. O folder com os respectivos manifests já esta presente [aqui](T2/Parte1/manifest).

  Requisitos:
  * Python >= 3.8
  * pip3 com os seguintes pacotes:
    * xmltodict
    * argparse

  Para executar a aplicação:
  ```
  python3 main.py --folder-name manifest/
  ```
  A segunda [Parte2](T2/Parte2), consiste de dois códigos `main.py` que deve receber um único arquivo executável como entrada e `main2.py` que deve receber dois arquivos executáveis.

  Requisitos:
  * Python >= 3.8
  * pip3 com os seguintes pacotes:
    * pefile
    * argparse

  Para executar a aplicação:
  ```
  python3 main.py --file-one Tibia_Setup.exe

  python3 main2.py --file-one Tibia_Setup.exe --file-two avira_pt-br_sptl1_89fae122b00d581b__pavwws.exe
  ```

### T3 ###

### T4 ###

### Projeto da disciplina ###

  Grupo: Tiago Heinrich [h31nr1ch](https://github.com/h31nr1ch) e Rodrigo Lemos [Rodrigo-Lemos](https://github.com/Rodrigo-Lemos)

  A proposta a ser estudada, consiste na utilização do dataset Androzoo junto com um dataset dinâmico (androDi) para a identificação de ataques no Android. Este dataset híbrido irá explorar características do androzoo (dataset estático) para solucionar problemáticas já identificadas com o androDi.

  O androDi foi explorado com o WEKA para a identificação de alguns comportamentos e melhor compreensão da distribuição do conjunto de dados. Permitindo identificar a importância e peso que features como BC_REPLY_SG, BC_REPLY e BR_TRANSACTION_COMPLETE tem sobre o dataset, também destacando atributos com baixa distribuição dentre as classes presentes.

  Os dados consistem de um conjunto de características dinâmicas extraídas de inúmeras execuções de aplicações no Android. As aplicações foram escolhidas tendo como base o dataset androzoo, ao qual já destaca um conjunto de características de aplicativos do Android. Esta representação dinâmica é ideal para identificação de ataques em tempo real, ao qual destaca uma das finalidades do modelo a ser proposto para realizar a identificação de ataques através de técnicas de machine learning.

  Especificamente os dados encontrados no androDi são numéricos, aos quais as 37 features destacam uma representação de aparições em um traço. Uma amostra é apresentada a seguir:

  ```
  Apk,Label,BC_REPLY_SG,BC_TRANSACTION,BC_REPLY,BC_ACQUIRE_RESULT,BC_FREE_BUFFER,BC_INCREFS,BC_ACQUIRE,BC_RELEASE,BC_DECREFS,BC_INCREFS_DONE,BC_ACQUIRE_DONE,BC_ATTEMPT_ACQUIRE,BC_REGISTER_LOOPER,BC_ENTER_LOOPER,BC_EXIT_LOOPER,BC_REQUEST_DEATH_NOTIFICATION,BC_CLEAR_DEATH_NOTIFICATION,BC_DEAD_BINDER_DONE,BC_TRANSACTION_SG,BR_ERROR,BR_OK,BR_TRANSACTION,BR_ACQUIRE_RESULT,BR_DEAD_REPLY,BR_TRANSACTION_COMPLETE,BR_INCREFS,BR_ACQUIRE,BR_RELEASE,BR_DECREFS,BR_ATTEMPT_ACQUIRE,BR_NOOP,BR_SPAWN_LOOPER,BR_FINISHED,BR_DEAD_BINDER,BR_CLEAR_DEATH_NOTIFICATION_DONE,BR_FAILED_REPLY,BR_REPLY
  com.ForntYardIdeas.eshall,1,9876,35739,21682,0,82659,4845,4854,4333,4329,3683,3684,0,38,12,0,605,574,19,15458,0,0,51074,0,0,82741,3683,3682,3485,3492,0,0,37,0,19,574,0,31615
  ```

  Devido ao tipo de dado e tipo de processo esperamos utilizar o AndroDi junto com uma seleção de features ainda não definidas do androzoo para aperfeiçoar a identificação de atividades maliciosas em um ambiente Android.

  Uma seleção de características já presentes no androzoo será utilizada para complementar a base dinâmica que consistem de chamadas de sistemas capturadas no binder do android, esta sendo representações de execuções de cada aplicação.

  Na parte exploratória do AndroDi foi constatado alguns atributos que só possuem uma única classe, como BR_ACQUIRE_RESULT e BR_DEAD_REPLY. Estas classes não apresentam nenhuma contribuição ao conjunto de dados e serão removidas.

  ![alt text](https://github.com/h31nr1ch/CDadosSeg/blob/main/TrabalhoFinal/img/weka.png?raw=true)
