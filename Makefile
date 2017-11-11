sample:
	rm -rf ./sample_dir
	rm -rf ./model_dir
	mkdir ./sample_dir
	mkdir ./model_dir
	python3 font2img.py --dst_font fonts/NotoSansCJK.ttc --src_font fonts/NotoSerifCJK.ttc --canvas_size 64  \
	--char_size 56 --x_offset 0 --y_offset -10 --shuffle 1 --mode L
	python3 img2pickle.py --dir sample_dir --save_dir model_dir

tb:
	rm -rf board/* && tensorboard --logdir=board

clean:
	rm -rf ./model_dir/*.png

zi2ziu_prepare:
	rm -rf ./zi2ziu_experiment
	mkdir -p ./zi2ziu_sample
	mkdir -p ./zi2ziu_data
	mkdir -p ./zi2ziu_experiment
	python3 font2img.py --src_font fonts/NotoSansCJK.ttc --dst_font fonts/NotoSerifCJK.ttc --sample_dir zi2ziu_sample --mode RGB
	python3 img2pickle.py --dir zi2ziu_sample --save_dir zi2ziu_data
	mkdir -p zi2ziu_experiment
	mv -f zi2ziu_sample zi2ziu_experiment/
	mv -f zi2ziu_data zi2ziu_experiment/data
	cp -f font27/* zi2ziu_experiment/

zi2ziu_train:
	python3 model/zi2ziU.py --experiment_dir zi2ziu_experiment --batch_size 16 --freeze_encoder 0 --lr 0.001

zi2ziu_clean:
	rm -rf zi2ziu_experiment/logs
	rm -rf zi2ziu_experiment/checkpoints