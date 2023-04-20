FROM efficientnet-b0_server:latest
WORKDIR /app
COPY . .
ENV FLASK_APP flask_server.py
ENTRYPOINT ["flask", "run", "--port=8000"]