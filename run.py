import sys
sys.path.append('/home/mich/Projects/food_calorie_estimation/modules')
import modules.image_segmentation as imseg
import modules.volume_estimation as ve
import cv2
import pandas as pd


# fx/fy ogniskowa kamery (iphone 13)
fx = 3080.56
fy = 3080.80
# cx/cy punkt główny kamery (iphone 13)
cx = 1543.23
cy = 1982.34


img_path = "media/media/kotlet01.jpg"

img = cv2.imread(img_path)

pred_mask = imseg.get_pred_mask(img)

volumes, meal_ids = ve.predict_volume(img, pred_mask,fx,fy,cx,cy, visualize=False)

print("")
df = pd.read_csv("tabela_kalorycznosci.csv")
sum = df.iloc[0, 3:] * 0
for volume, id in zip(volumes, meal_ids):
    weight = volume * df.iloc[id-1, 2] / 100
    print(f"{df.iloc[id-1, 1]} - {weight*100:.2f} g")
    print(df.iloc[id-1, 3:] * weight)
    sum += df.iloc[id-1, 3:] * weight
    print("")

print("Łącznie:")
print(sum)
