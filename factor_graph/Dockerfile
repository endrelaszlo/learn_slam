FROM --platform=linux/amd64 ubuntu:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh
#RUN /bin/bash ~/miniconda.sh -b -p /opt/conda
#RUN rm ~/miniconda.sh 
#RUN /opt/conda/bin/conda clean -tipsy 
#RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh 
#RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
#RUN echo "conda activate base" >> ~/.bashrc

RUN apt-get update --fix-missing && \
    apt-get install -y libsuitesparse-dev
COPY . /src
WORKDIR /src
RUN conda env create -f environment.yml

RUN apt-get install -y vim tmux

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

