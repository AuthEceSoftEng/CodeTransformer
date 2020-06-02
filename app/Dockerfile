FROM python:3.6
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["gunicorn", "main:app", "-b", ":8080", "-t", "1000"]