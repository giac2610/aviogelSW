# Aviogel Software

E' possibile trovare l'intera documentazione [qui](https://docs.google.com/document/d/1HzIDs8MNS6pjdDcSqwxV5OlhAUNFD5ezyQfqX5_XvMQ/edit?tab=t.0)

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
