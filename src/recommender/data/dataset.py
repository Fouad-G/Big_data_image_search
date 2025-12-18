from pathlib import Path
import cv2
from typing import Iterator, Tuple


#hier wird ein generator erstellt um ein Bild mit id zu yielden
#rglob , findet alle bilder die mit jpg sind egal wie tief rekursiv im ordner

def image_generator(root_dir: Path,)-> Iterator[Tuple[int,any]]:
    image_id=0
    for img_path in root_dir.rglob("*.jpg"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        yield image_id, image
        image_id +=1