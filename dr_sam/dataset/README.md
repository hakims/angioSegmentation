## Dataset

Dataset consists of carefully selected images from 500 angiographic examinations of the pelvic-iliac arteries, carried out between 2018 and 2024 at Bad Oeynhausen Hospital and JWK Klinikum Minden, 
within their radiology departments. The focus of these examinations was the abdominal aorta below the renal arteries and the pelvic arteries. Of these images, 450 have a resolution of 386x448 pixels, and 50 have a resolution of 819x950 pixels. The dataset includes 170 images featuring at least one stenosis and 64 images with at least one aneurism.
The dataset archive is organized as follows:

- `images/`: Contains X-ray angiography images. Each image represents `{image_id}.jpg` 
- `masks/`: Contains binary masks. Each masks represents `{image_id}.png`.
- `metadata.json`: Contains information about the bounding boxes and the anomalies in the images.
- `README.md`: Contains information about the dataset.

`metadata.json`: This file contains information about bounding boxes and anomalies in the images. The file is structured as follows:

```json5
  {
    "image_id": 1, // Image id in the dataset
    "anomalies": { // Anomalies in the image with the format {anomaly: [(x, y)]}
      "stenosis": [[270, 351]], // Coordinates of the stenosis, if no stenosis, the field is empty []
      "aneurysm": [[229, 388], [253, 361]] // Coordinates of the aneurism, if no aneurism, the field is empty []
    },
    "bboxes": [[158, 11, 134, 192], 
               [13, 175, 187, 256],
               [193, 147, 159, 246]] // Bounding boxes of the vessels of interest with the format [x, y, width, height]
  }
```