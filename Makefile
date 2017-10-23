sample:
	rm -rf ./sample_dir
	mkdir ./sample_dir
	python3 font2img.py --src_font fonts/NotoSansCJK.ttc --dst_font fonts/NotoSerifCJK.ttc --canvas_size 64  \
	--char_size 56 --x_offset 0 --y_offset -10