import os
import time
import json
import pandas as pd
import paho.mqtt.client as mqtt

# MQTT Configurations
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = "sentinel/telemetry/"

# Load a sample cleaned CSV to simulate logs
cleaned_file_path = r"/home/ubuntu/sentinel_xdr/cleaned_datasets/cleaned_IoT_GPS_Tracker.csv"
model_name = "IoT_GPS_Tracker"

print("Starting MQTT Telemetry Streamer...")

# Connect to Mosquitto Broker
client = mqtt.Client()
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
except Exception as e:
    print(f"Error: Could not connect to Mosquitto broker at {MQTT_BROKER}:{MQTT_PORT}.")
    print("Ensure Mosquitto is running: 'sudo systemctl start mosquitto'")
    exit(1)

client.loop_start()

if not os.path.exists(cleaned_file_path):
    # Fallback to current folder if relative
    cleaned_file_path = "cleaned_datasets/cleaned_IoT_GPS_Tracker.csv"

if not os.path.exists(cleaned_file_path):
    print(f"Error: Cleaned CSV file not found at: {cleaned_file_path}")
    print("Please copy your cleaned datasets to the VM.")
    exit(1)

df = pd.read_csv(cleaned_file_path)
X_feats = df.drop(columns=['label', 'type'])

print(f"Loaded {len(X_feats)} logs for streaming on topic: {MQTT_TOPIC_PREFIX}{model_name}")

try:
    for idx, row in X_feats.iterrows():
        # Prepare the log payload
        payload = {
            "device_id": f"gps_device_{idx % 3}",
            "features": row.values.tolist(),
            "source_ip": f"192.168.1.{100 + (idx % 10)}",
            "process_id": 2000 + idx
        }
        
        # Publish
        topic = f"{MQTT_TOPIC_PREFIX}{model_name}"
        client.publish(topic, json.dumps(payload))
        
        print(f"[{idx}] Published log event to {topic}")
        
        # Sleep to simulate real-time ingestion (2 events per second)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStream stopped by user.")

client.loop_stop()
client.disconnect()
print("Disconnected successfully.")
