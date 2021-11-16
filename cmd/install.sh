#!/bin/bash

PY_VERSION="python3.8"

if [ -z "$SUDO_UID" ]; then
  echo "Using with sudo is required" && exit 1
fi

if [ 0 -eq "$SUDO_UID" ]; then
  echo "Don't exec this script by root" && exit 1
fi

if [ -n "$($PY_VERSION -m venv --help 1>/dev/null 2>/dev/null || echo $?)" ]; then
  if [ 0 -ne "$(id -u)" ]; then
    echo "Needs sudo-permissions to install 'python3.8-venv'" && exit 1
  fi
  while true; do
    echo
    read -rp "Install 'python3.8-venv'? y/n: " yn
    case $yn in
        [Yy]* ) apt install -y python3.8-venv; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
  done
fi

PROJ_DIR=$(dirname "$(dirname "$0")")
VENV_PATH="$PROJ_DIR/venv"
while [ ! -d "$VENV_PATH" ]; do
    read -rp "Install venv to '$VENV_PATH'? y/n: " yn
    case $yn in
        [Yy]* ) su "$SUDO_USER" -pc "$PY_VERSION -m venv $VENV_PATH"; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

su "$SUDO_USER" -pc \
    ". $VENV_PATH/bin/activate ; pip install --no-cache-dir -r $PROJ_DIR/requirements.txt"
