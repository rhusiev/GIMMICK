import pandas as pd

CHUNKS_IN_REGION = 1024

chunk_nums = pd.read_csv("chunk_nums.csv")
times = pd.read_csv("times.csv")

mc_chunks = chunk_nums["mc"].iloc[0]
gimmick_chunks = chunk_nums["gimmick"].iloc[0]

mc_avg_time = times["mc"].mean()
gimmick_avg_time = times["gimmick"].mean()

mc_time_per_chunk = (mc_avg_time / mc_chunks) * 1000
gimmick_time_per_chunk = (gimmick_avg_time / gimmick_chunks) * 1000

mc_time_per_region = CHUNKS_IN_REGION * mc_time_per_chunk
gimmick_time_per_region = CHUNKS_IN_REGION * gimmick_time_per_chunk

print(f"MC: {mc_time_per_region:.4f} ms/region")
print(f"Gimmick: {gimmick_time_per_region:.4f} ms/region")
