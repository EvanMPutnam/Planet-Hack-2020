import os
import sys
import json

from PIL import Image

def generate_annotations(path):
    lst = []
    for root, dirs, files in os.walk(path):
        for fle in files:
            im = Image.open(os.path.join(root, fle))
            width, height = im.size
            items = {
                "image": fle,
                "annotations": [
                    {
                        "label": "baseball_field",
                        "coordinates": {
                            "x": 0,
                            "y": 0,
                            "width": width,
                            "height": height
                        }
                    }
                ]
            }
            lst.append(items)
            im.close()

    to_json = json.dumps(lst, indent=4)
    with open(path + "/annotations.json", "w+") as fle: fle.write(to_json)

generate_annotations("../bbf_train")




