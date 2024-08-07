#FROM arm64v8/python:3.11-slim
FROM python:3.8-slim
WORKDIR /app/python-app

COPY . .

RUN echo '\nAcquire::http::Proxy "http://autoproxy.ktag.ch:8080";' >> /etc/apt/apt.conf \ 
	&& apt-get update \ 
	&& apt-get -y install tesseract-ocr-deu \ 
	&& apt-get -y install git-all \ 
	&& apt-get -y install g++ \ 
	&& pip3 install --upgrade pip --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install -q pyyaml==5.1 --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install -q torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html --proxy http://autoproxy.ktag.ch:8080 --trusted-host download.pytorch.org \ 
	&& pip3 install 'git+https://github.com/facebookresearch/detectron2.git' --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install -r requirements.txt --proxy http://autoproxy.ktag.ch:8080
  
CMD ["python3", "loop.py"]