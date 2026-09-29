FROM python:3.13

# create new user vscode
RUN useradd -ms /bin/bash teli

# change WORKDIR
WORKDIR /home/teli

# switch User 
USER teli

# copy packages.txt file
RUN mkdir Downloads
COPY requirements.txt ./Downloads
RUN pip3 install -r ./Downloads/requirements.txt

# change WORKDIR
RUN mkdir workspace
WORKDIR /home/teli/workspace

# Set PYTHONUNBUFFERED
ENV PYTHONUNBUFFERED=1

# CMD command
CMD ["/bin/bash"]
