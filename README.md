# CDadosSeg #
 Resolução atividades ci1030-ERE2-CiênciaDeDados (source completo no repositório privado cds-ufpr)

## Índice :floppy_disk: ##
- [Para que serve este repositório?](#Para-que-serve-este-repositório?)
- [Atividades](#Atividades)
    - [T2](#T2)
    - [T3](#T3)
    - [T4](#T4)

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
    python3 main.py --file-one ../../../pratica-3/source/Tibia_Setup.exe

    python3 main2.py --file-one Tibia_Setup.exe --file-two avira_pt-br_sptl1_89fae122b00d581b__pavwws.exe
    ```

### T3 ###

### T4 ###
