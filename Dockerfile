FROM python:3.7
WORKDIR /Plagpolice
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD [ "python3", "-m" , "app"]