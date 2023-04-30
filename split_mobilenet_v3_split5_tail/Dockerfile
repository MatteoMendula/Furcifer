FROM efficientnet-b0_server:latest
WORKDIR /app
COPY . .
ENV FLASK_APP server.py
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT 8001
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
EXPOSE 8001
# RUN echo "The value of MY_VAR is: $PORT"
# ENTRYPOINT [ "echo", "${PORT}" ]
ENTRYPOINT ["flask", "run"]