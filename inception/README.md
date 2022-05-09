# How to run Inception recognition?

```bash
python scripts/label_image.py --image image.png
```


# How to run Inception retraining?

```bash
python scripts/retrain.py --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --image_dir=tf_files/flower_photos
```





LITERATURE: https://towardsdatascience.com/training-inception-with-tensorflow-on-custom-images-using-cpu-8ecd91595f26

ADDITIONAL LITERATURE:
https://github.com/sourcedexter/tfClassifier
https://sourcedexter.com/quickly-setup-tensorflow-image-recognition/
https://sourcedexter.com/retrain-tensorflow-inception-model/
https://sourcedexter.com/tensorflow-text-classification-python/