FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.14.1
WORKDIR /bot
COPY . .
RUN python3 worldmap.py /bot/arena /bot/mapcache.npz && chmod -R a+rX /bot
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
