python3 asr_eval.py --signs --p_dir blue_low  --model frcnn --weights WEIGHTS/FRCNN_blue_low_morphing.pt
python3 asr_eval.py --signs --p_dir blue_high --model frcnn --weights WEIGHTS/FRCNN_blue_high_morphing.pt
python3 asr_eval.py --signs --p_dir blue_both --model frcnn --weights WEIGHTS/FRCNN_blue_multi_morphing.pt

