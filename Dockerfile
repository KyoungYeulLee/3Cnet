FROM python:3.8.18-bullseye

RUN mkdir 3Cnet & pip install --upgrade pip
COPY . 3Cnet
RUN pip install -r 3Cnet/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
