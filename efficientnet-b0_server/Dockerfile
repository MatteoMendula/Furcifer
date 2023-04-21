FROM efficientnet-b0_server:latest
WORKDIR /app
COPY . .
ENV FLASK_APP flask_server.py
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT 8000
# RUN echo "The value of MY_VAR is: $PORT"
# ENTRYPOINT [ "echo", "${PORT}" ]
ENTRYPOINT ["flask", "run"]