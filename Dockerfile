# parent image
FROM python:3.8-slim

WORKDIR /app/python-app

COPY . .

# install FreeTDS and dependencies
RUN echo '\nAcquire::http::Proxy "http://autoproxy.ktag.ch:8080";' >> /etc/apt/apt.conf \  
	&& apt-get update \
	&& apt-get -y install unixodbc \
	&& apt-get -y install unixodbc-dev \
	&& apt-get -y install freetds-dev \
	&& apt-get -y install freetds-bin \
	&& apt-get -y install tdsodbc \
	&& apt-get -y install --reinstall build-essential \
	&& apt-get -y install tesseract-ocr-deu \ 
	&& apt-get -y install git-all \ 
	&& apt-get -y install g++

# populate "ocbcinst.ini"
RUN echo "[FreeTDS]\n\
Description = FreeTDS unixODBC Driver\n\
Driver = /usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so\n\
Setup = /usr/lib/x86_64-linux-gnu/odbc/libtdsS.so" >> /etc/odbcinst.ini

# install pyodbc (and, optionally, sqlalchemy)
RUN pip3 install pyodbc==4.0.26 sqlalchemy==1.3.5 --proxy http://autoproxy.ktag.ch:8080 --trusted-host pypi.python.org \
	&& pip3 install --upgrade pip --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install -q pyyaml==5.1 --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install -q torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html --proxy http://autoproxy.ktag.ch:8080 --trusted-host download.pytorch.org \ 
	&& pip3 install 'git+https://github.com/facebookresearch/detectron2.git' --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install sentencepiece --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install accelerate -U --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install pytesseract --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install transformers --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install datasets --proxy http://autoproxy.ktag.ch:8080 \ 
	&& pip3 install -r requirements.txt --proxy http://autoproxy.ktag.ch:8080

# run app.py upon container launch
CMD ["python3", "loop.py"]