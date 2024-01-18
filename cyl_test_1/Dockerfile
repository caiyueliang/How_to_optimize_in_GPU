FROM swr.cn-central-221.ovaijisuan.com/wair/pytorch_2_0_0:pytorch_2.0.0-cuda_11.8-py_3.9-ubuntu_18.04-x86_64-20230423
RUN pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple/

WORKDIR /server/test
COPY . .

CMD ["python", "flask_server.py"]

