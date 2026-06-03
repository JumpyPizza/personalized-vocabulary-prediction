# import os

# # folder containing your files
# folder = "vkd"

# # label mapping
# label_map = {"1": "0", "2":"0","3":"0","4":"1" ,"5": "1"}

# for filename in os.listdir(folder):
#     filepath = os.path.join(folder, filename)
    
#     # only process regular files
#     if not os.path.isfile(filepath):
#         continue
    
#     with open(filepath, "r", encoding="utf-8") as f:
#         lines = f.readlines()
    
#     new_lines = []
#     for line in lines:
#         line = line.strip()  # remove whitespace and newlines
#         if not line:  # skip empty lines
#             continue
#         if "\t" not in line:
#             continue  # skip malformed rows

#         word, label = line.split("\t", 1)
#         mapped_label = label_map.get(label, label)  # map if known
#         new_lines.append(f"{word}\t{mapped_label}")
    
#     # rewrite file without extra newline at end
#     with open(filepath, "w", encoding="utf-8") as f:
#         f.write("\n".join(new_lines))


import os

folder = "evkd_data"

for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if not os.path.isfile(filepath):
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))