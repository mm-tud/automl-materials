FROM mfeurer/auto-sklearn:master

RUN mkdir /data
RUN mkdir /workdir

WORKDIR /workdir

RUN pip3 install matplotlib

ENV OPENBLAS_NUM_THREADS=6
ENV MKL_NUM_THREADS=6
ENV OMP_NUM_THREADS=6

CMD python3 run.py
