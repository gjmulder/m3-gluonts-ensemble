#!/bin/sh

ssh heika sudo killall -u jenkins python3
sudo killall -u jenkins python3
ssh asushimu sudo killall -u jenkins /usr/bin/python3
ssh yokojitsu sudo killall -u jenkins /usr/bin/python3
