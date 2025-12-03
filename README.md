# Aviogel Software

## Documentazione

E' possibile trovare **l'intera documentazione** [qui](https://docs.google.com/document/d/1HzIDs8MNS6pjdDcSqwxV5OlhAUNFD5ezyQfqX5_XvMQ/edit?tab=t.0)

## TODO FUTURI POST GIACOMO

Se stai leggendo questo readme, vuol dire che sei il successore di colui che l'ha scritto, ti sto dando **un consiglio** dal passato per il futuro:

Sulla raspberry crea una repository di _deploy_, duplicando quella attuale che diventerà repo di _testing_.

**Perchè?**

Per evitare di compromettere eventualmente la versione di deploy nel caso venisse qualche investitore per una demo. Il bug insensato è sempre dietro l'angolo con aviogel.
Una volta testata quelli di testing si può aggiornare anche quella di deployment, ma prima devi essere molto convinto di non aver fatto danni.

## AVVIO

Se chiede la password, è a casusa del comando SUDO, quindi è il pin del pc

### macOS

dentro la cartella aviogelSW basta dare il comando

> sh startMacOS.sh

### windows

dentro la cartella aviogelSW basta fare doppio click su:

> startWin.bat

Non è mai stato testato però, potrebbe non funzionare

## Update aviogel on raspberry

### Update all

```shell
sh updateAviogel.sh
```

this one often need a reboot

```shell
sudo reboot now
```

### Update backend only

```shell
sh updateBackend.sh
```

## Coding tips

### Virtual Env

Avviare sempre il virtual Env per python

#### macOs

```shell
source .venv/bin/activate  
```

#### raspberry

```shell
cd aviogelSW/backend
source aviogelEnv/bin/activate
```

## OS image

[un file immagine](https://drive.google.com/file/d/1aDWKli4Vgne-sSIcQFIgW6Ttj_DsgID8/view?usp=drive_link) del sistema operativo da cui si può ripartire per poi aggiornare l'app, infatti una volta caricata l'immagine è possibile fare un aggiornamento alle ultime modifiche. Fare riferimento a [update all](#update-all)
