sample:
	rm -rf ./sample_dir
	rm -rf ./model_dir
	mkdir ./sample_dir
	mkdir ./model_dir
	python3 font2img.py --dst_font fonts/NotoSansCJK.ttc --src_font fonts/NotoSerifCJK.ttc --canvas_size 64  \
	--char_size 56 --x_offset 0 --y_offset -10 --shuffle 1 --mode L
	python3 img2pickle.py --dir sample_dir --save_dir model_dir

sample_xingkai:
	rm -rf ./sample_dir
	rm -rf ./model_dir
	mkdir ./sample_dir
	mkdir ./model_dir
	python3 font2img.py --src_font fonts/NotoSansCJK.ttc --dst_font fonts/XingKai.ttf --canvas_size 64  \
	--char_size 48 --x_offset 0 --y_offset 0 --shuffle 1 --mode L  --charset GB2312 --tgt_x_offset 0 --tgt_y_offset 5 --tgt_char_size 60 --sample_count 3000
	python3 img2pickle.py --dir sample_dir --save_dir model_dir

tb:
	rm -rf board/* && tensorboard --logdir=board

clean:
	rm -rf ./model_dir/*.png

install_icu:
	CFLAGS=-I/usr/local/opt/icu4c/include LDFLAGS=-L/usr/local/opt/icu4c/lib ICU_VERSION=57 pip install --user pyicu;
	pip install --user fonttools tabulate fontaine

zi2ziu_prepare:
	rm -rf ./zi2ziu_experiment
	mkdir -p ./zi2ziu_sample
	mkdir -p ./zi2ziu_data
	mkdir -p ./zi2ziu_experiment
	python3 font2img.py --src_font fonts/NotoSansCJK.ttc --dst_font fonts/NotoSerifCJK.ttc --sample_dir zi2ziu_sample --mode L
	python3 img2pickle.py --dir zi2ziu_sample --save_dir zi2ziu_data
	mkdir -p zi2ziu_experiment
	mv -f zi2ziu_sample zi2ziu_experiment/
	mv -f zi2ziu_data zi2ziu_experiment/data

zi2ziu_prepare_xingkai:
	rm -rf ./zi2ziu_experiment
	mkdir -p ./zi2ziu_sample
	mkdir -p ./zi2ziu_data
	mkdir -p ./zi2ziu_experiment
	python3 font2img.py --src_font fonts/NotoSansCJK.ttc --dst_font fonts/XingKai.ttf --sample_dir zi2ziu_sample \
	--x_offset 0 --y_offset 0 --shuffle 1 --mode L  --charset GB2312 --tgt_x_offset 0 --tgt_y_offset 20 --tgt_char_size 180 --sample_count 3000
	python3 img2pickle.py --dir zi2ziu_sample --save_dir zi2ziu_data
	mkdir -p zi2ziu_experiment
	mv -f zi2ziu_sample zi2ziu_experiment/
	mv -f zi2ziu_data zi2ziu_experiment/data

zi2ziu_prepare_xingkai_randompair:
	rm -rf ./zi2ziu_experiment_randompair
	mkdir -p ./zi2ziu_sample
	mkdir -p ./zi2ziu_data
	mkdir -p ./zi2ziu_experiment_randompair
	python3 font2img_randompair.py --src_font fonts/NotoSansCJK.ttc --dst_font fonts/XingKai.ttf --sample_dir zi2ziu_sample \
	--x_offset 10 --y_offset 0 --shuffle 1 --mode L  --charset GB2312 --tgt_x_offset 0 --tgt_y_offset 20 --tgt_char_size 180 --sample_count 3300 --overlap 0
	python3 img2pickle.py --dir zi2ziu_sample --save_dir zi2ziu_data
	mkdir -p zi2ziu_experiment_randompair
	mv -f zi2ziu_sample zi2ziu_experiment_randompair/
	mv -f zi2ziu_data zi2ziu_experiment_randompair/data


zi2ziu_prepare_xiaozhuan:
	rm -rf ./zi2ziu_experiment_xiaozhuan
	rm -rf ./zi2ziu_sample
	mkdir -p ./zi2ziu_sample
	mkdir -p ./zi2ziu_data
	mkdir -p ./zi2ziu_experiment_xiaozhuan
	python3 font2img_randompair.py --src_font fonts/NotoSansCJK.ttc --dst_font fonts/XiaoZhuan.ttf --sample_dir zi2ziu_sample \
	--x_offset 0 --y_offset 0 --shuffle 1 --mode L  --charset GB2312 --tgt_x_offset -30 --tgt_y_offset 20 --tgt_char_size 220 --sample_count 3300 --overlap 1
	python3 img2pickle.py --dir zi2ziu_sample --save_dir zi2ziu_data
	mkdir -p zi2ziu_experiment_xiaozhuan
	mv -f zi2ziu_data zi2ziu_experiment_xiaozhuan/data

zi2ziu_train:
	python3 model/zi2ziU.py --experiment_dir zi2ziu_experiment --batch_size 16 --freeze_encoder 0 --lr 0.001 --Ltv_penalty 0.001

zi2ziu_train_xingkai:
	python3 model/zi2ziU.py --experiment_dir ../mount/xingkai_exp --batch_size 16 --freeze_encoder 0 --lr 0.001 --Ltv_penalty 0.001 --image_size 256


zi2ziu_train_xiaozhuan:
	python3 model/zi2ziU.py --experiment_dir ../mount/zi2ziu_experiment_xiaozhuan/ --batch_size 16 --freeze_encoder 0 --lr 0.001 --Ltv_penalty 0.001 --image_size 256

zi2ziu_train_xingkai_randompair:
	python3 model/zi2ziU.py --experiment_dir ../mount/zi2ziu_experiment_randompair/ --batch_size 16 --freeze_encoder 0 --lr 0.001 --Ltv_penalty 0.001 --image_size 256

zi2ziu_train_xingkai_randompair_augment:
	python3 model/zi2ziU.py --experiment_dir ../mount/zi2ziu_experiment_randompair_augment/ --batch_size 16 --freeze_encoder 0 --lr 0.001 --Ltv_penalty 0.001 --image_size 256 --augment

zi2ziu_clean:
	rm -rf zi2ziu_experiment/logs
	rm -rf zi2ziu_experiment/checkpoints