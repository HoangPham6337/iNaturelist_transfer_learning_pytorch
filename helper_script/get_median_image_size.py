import os
import statistics

all_file_sizes = []

top_level_dir = "./data/haute_garonne"

for dirpath, dirnames, filenames in os.walk(top_level_dir):
    for file_name in filenames:
        if file_name.endswith("parquet") or file_name.endswith("json"):
            continue
        full_path = os.path.join(dirpath, file_name)
        try:
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                all_file_sizes.append(size)
        except OSError as e:
            print(e)
            pass

print(len(all_file_sizes))
avg = statistics.mean(all_file_sizes)
print(f"{avg:.3f} byte")
print(f"{avg / 1024:.3f} kilobyte")