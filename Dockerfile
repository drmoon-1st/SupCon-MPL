FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY requirement.txt .

RUN pip install -r requirement.txt

RUN rm requirement.txt

CMD ["bash"]