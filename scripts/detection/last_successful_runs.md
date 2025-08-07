## compare_local_models.py

python scripts/detection/compare_local_models.py --model1 /Users/abhinavrai/Playground/snooker_data/trained models/snkr_segm-egvem-2-yolov11-weights.pt --model2 /Users/abhinavrai/Playground/snooker_data/trained models/snkr_segm-egvem-3-roboflow-weights.pt --input-dir /Users/abhinavrai/Playground/snooker/test_images --output-dir /tmp/compare_local_inference/segm-egvem-2-yolov11_vs_segm-egvem-3-roboflow
Loading model 1: /Users/abhinavrai/Playground/snooker_data/trained models/snkr_segm-egvem-2-yolov11-weights.pt
Loading model 2: /Users/abhinavrai/Playground/snooker_data/trained models/snkr_segm-egvem-3-roboflow-weights.pt
Loading model 1: snkr_segm-egvem-2-yolov11-weights.pt
Loaded model from /Users/abhinavrai/Playground/snooker_data/trained models/snkr_segm-egvem-2-yolov11-weights.pt
Loading model 2: snkr_segm-egvem-3-roboflow-weights.pt
Loaded model from /Users/abhinavrai/Playground/snooker_data/trained models/snkr_segm-egvem-3-roboflow-weights.pt
Found 7 images to process
Processing images: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  4.16it/s]

==================================================
Processing complete!
Total images processed: 7
Total time: 1.70 seconds
Average time per image: 0.24 seconds

Per-model inference performance:
  snkr_segm-egvem-2-yolov11-weights:
      Average inference time: 172.56 ms
          Inference it/sec: 5.80
            snkr_segm-egvem-3-roboflow-weights:
                Average inference time: 36.36 ms
                    Inference it/sec: 27.50

                    Output images saved to: /tmp/compare_local_inference/segm-egvem-2-yolov11_vs_segm-egvem-3-roboflow
                    ==================================================

