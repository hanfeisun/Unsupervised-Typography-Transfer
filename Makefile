sample:
	rm -rf ./sample_dir
	rm -rf ./model_dir
	mkdir ./sample_dir
	mkdir ./model_dir
	python3 font2img.py --src_font fonts/NotoSansCJK.ttc --dst_font fonts/NotoSerifCJK.ttc --canvas_size 64  \
	--char_size 56 --x_offset 0 --y_offset -10 --shuffle 1
	python3 img2pickle.py --dir sample_dir --save_dir model_dir

tb:
	rm board/* && tensorboard --logdir=board