FROM python:3.8.18-bullseye

RUN mkdir cccnet & pip install --upgrade pip
COPY . cccnet
RUN pip install -r cccnet/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
