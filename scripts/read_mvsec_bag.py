import rosbag
from tqdm import tqdm
from collections import defaultdict

bag_path = '/home/richeek/Downloads/outdoor_night1_data.bag'
topic_ctr = defaultdict(int)

with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in tqdm(bag.read_messages(), desc="Reading bag file"):
        topic_ctr[topic] += 1

print("Message counts per topic:")
for topic, count in topic_ctr.items():
    print(f"{topic}: {count}")
print("#" + "-" * 40 + "# \n")
print("Bag file read successfully.")