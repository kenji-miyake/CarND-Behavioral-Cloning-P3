@echo off
cd %~dp0

docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v %~dp0:/src udacity/carnd-term1-starter-kit
