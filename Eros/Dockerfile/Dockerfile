FROM tensorflow/tensorflow:2.8.0-gpu

USER ${NB_UID}

RUN pip install --no-cache-dir --upgrade pip && \
    pip install opencv-python-headless

RUN pip install pandas
RUN pip install tensorflow-addons
RUN pip install vit-keras
RUN pip install matplotlib
RUN pip install sklearn
RUN pip install einops
RUN pip install -U efficientnet
RUN pip install tensorflow-datasets
