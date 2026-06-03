import pandas as pd 
import json
evkd = pd.read_csv("EVKD_lrec2018.csv", encoding = "cp932")
user_map = {}
for idx, row in enumerate(evkd.iloc):
    print(row)
    if row.TOEICscore <600:
        user_map[idx] = "beginner"
    elif row.TOEICscore <800:
        user_map[idx] = "intermediate"
    else:
        user_map[idx] = "advanced"

with open("evkd_user_map.json", "w", encoding="utf-8") as f:
    json.dump(user_map, f)
# print(user_map)