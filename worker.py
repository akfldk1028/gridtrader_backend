import dramatiq
import redis

# Configure the broker
broker = redis.Redis(host='localhost', port=6379, db=0)
dramatiq.set_broker(broker)

if __name__ == "__main__":
    dramatiq.Worker().run()