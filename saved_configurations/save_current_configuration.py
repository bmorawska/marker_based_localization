import pickle
import os
from datetime import datetime

from localization.real_values import real_values

configuration_name = "piwnica"
if not os.path.exists(configuration_name):
    os.makedirs(configuration_name)

metadata_file = 'metadata.txt'
with open(os.path.join(configuration_name, 'metadata.txt'), 'w') as f:
    now = datetime.now()
    date_string = now.strftime("%d/%m/%Y %H:%M:%S")
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    f.write(f"author: Barbara Morawska\n")
    f.write(f"configuration name: {configuration_name}\n")
    f.write(f"datetime: {date_string}\n")
    f.write("\n")
    f.write("# Pixel order\n")
    f.write("# 1------------------------0\n")
    f.write("# |                        |\n")
    f.write("# |                        |\n")
    f.write("# |                        |\n")
    f.write("# |                        |\n")
    f.write("# |                        |\n")
    f.write("# |                        |\n")
    f.write("# |                        |\n")
    f.write("# 2------------------------3\n")
    f.write("\n")
    f.write("ArUco map:\n")
    for key in real_values:
        f.write(f"\t{key}:\n")
        f.write(f"\t\t{real_values[key][0]}\n")
        f.write(f"\t\t{real_values[key][1]}\n")
        f.write(f"\t\t{real_values[key][2]}\n")
        f.write(f"\t\t{real_values[key][3]}\n")


with open(os.path.join(configuration_name, 'map.pickle'), 'wb') as handle:
    pickle.dump(real_values, handle, protocol=pickle.HIGHEST_PROTOCOL)


