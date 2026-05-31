FROM python:3.8.18-bullseye

RUN mkdir -p 3Cnet && python -m pip install --upgrade pip
COPY . 3Cnet
RUN python -m pip install -r 3Cnet/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
