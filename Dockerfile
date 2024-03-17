FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV env=docker
COPY requirements.txt /tva/requirements.txt
COPY tva.ipynb /tva/tva.ipynb
COPY src/ /tva/src/
COPY run_tva.sh .
RUN mkdir output/

RUN apt-get update && \
    apt-get install -y \
    python3  \
    python3-pip \
    texlive-xetex \
    pandoc
RUN pip install -r /tva/requirements.txt

RUN chmod +x run_tva.sh
ENTRYPOINT ["./run_tva.sh"]
