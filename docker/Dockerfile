FROM pytorch/pytorch AS deepvo
RUN apt-get update \
    && apt-get --assume-yes install \
        vim \
        git \
        wget
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python && pip install scipy && pip install torchvision
RUN git clone https://github.com/utiasSTARS/pykitti.git /pykitti
RUN cd /pykitti && python setup.py install && cd /
ENTRYPOINT /bin/bash