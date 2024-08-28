python3 asr_eval.py --signs --weights WEIGHTS/YOLO_blue_low_morphing.pt   --p_dir blue_low  --model yolo
python3 asr_eval.py --signs --weights WEIGHTS/YOLO_blue_high_morphing.pt  --p_dir blue_high --model yolo
python3 asr_eval.py --signs --weights WEIGHTS/YOLO_blue_multi_morphing.pt --p_dir blue_both --model yolo