FROM uber/horovod:0.14.0-tf1.10.0-torch0.4.0-py3.5

RUN apt-get -y update

RUN pip3 install mpi4py psutil gpustat scikit-optimize h5py==2.7.0 && \
    ln -s /usr/bin/python3.5 /usr/bin/python3

RUN apt-get -y install s3cmd

RUN git clone "https://github.com/svalleco/mpi_learn.git" "/mpi_learn"

WORKDIR "/mpi_learn"
