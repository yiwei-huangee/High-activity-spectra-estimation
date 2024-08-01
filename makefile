# RUN_TEST_WORK_FC: Test model with signal source and compare with fast correction algorithm
# RUN_TEST_WORK_MULTI_FC: Test model with multi source and compare with fast correction algorithm
# RUN_TEST_WORK: Test model with signal source
# RUN_TEST_WORK_MULTI: Test model with multi source
CPPDIR = ./traditional_methods/fast_correction

define RUN_TEST_WORK_FC
	# 1: duration 2:source 3:lambda_sim 4: lambda_fast_correction
	echo "$(1)_$(2)_$(3)_$(4)"
	cd $(CPPDIR)/src \
	&& g++ simulation_spectre_3D.cpp -o simulation -lgsl -lgslcblas \
	&& ./simulation ../acsfiles/densite_3d_pour_simulation4096_$(1)_$(2)_$(3).asc ../results/spectre_3d_temps_energie.csv $(4) \
	&& g++ deconvolution3Dfast.cpp -o deconvolution3Dfast \
	&& ./deconvolution3Dfast ../results/ds2Dfast_$(1)_$(2)_$(3).csv ../results/ds3Dfast_$(1)_$(2)_$(3).csv
    python main.py --run_mode='test'  --bins=1024 --source="{'name':'$(2)','weights':1}" --batch_size=16 --test_lambda_n=$(3)
endef
define RUN_TEST_WORK_MULTI_FC
	# 1: duration 2:source1 3:source2 4:lambda_sim 5: lambda_fast_correction
	# echo "$(1)_['$(2)','$(3)']_$(4)"
	cd $(CPPDIR)/src \
	&& g++ simulation_spectre_3D.cpp -o simulation -lgsl -lgslcblas \
	&& ./simulation ../acsfiles/"densite_3d_pour_simulation4096_$(1)_['$(2)','$(3)']_$(4).asc" ../results/spectre_3d_temps_energie.csv $(5) \
	&& g++ deconvolution3Dfast.cpp -o deconvolution3Dfast \
	&& ./deconvolution3Dfast ../results/"ds2Dfast_$(1)_['$(2)','$(3)']_$(4).csv" ../results/"ds3Dfast_$(1)_['$(2)','$(3)']_$(4).csv"
    python main.py --run_mode='test'  --bins=1024 --source="{'name':['$(2)','$(3)'],'weights':[1,2]}" --batch_size=16 --test_lambda_n=$(4)
endef
define RUN_TEST_WORK
	python main.py --run_mode='test'  --bins=1024 --source="{'name':'$(1)','weights':1}" --batch_size=16 --test_lambda_n=$(2)

endef
define RUN_TEST_WORK_MULTI
	python main.py --run_mode='test'  --bins=1024 --source="{'name':['$(1)','$(2)'],'weights':[1,2]}" --batch_size=16 --test_lambda_n=$(3)

endef
define RUN_TRAIN_WORK
	# 1:source 2:lambda_n
	python main.py --run_mode='train' --source="{'name':'$(1)','weights':1}" --train_lambda_n=$(2)
endef
define RUN_TRAIN_WORK_SNR
	# 1:source 2:lambda_n 3.snr
	python main.py --run_mode='train' --source="{'name':'$(1)','weights':1}" --train_lambda_n=$(2) --noise_unit='snr' --noise=$(3)
endef
train_Ac-225_0.08:
	$(call RUN_TRAIN_WORK,Ac-225,0.08)
train_Am-241_0.08:
	$(call RUN_TRAIN_WORK,Am-241,0.08)
train_Ba-131_0.08:
	$(call RUN_TRAIN_WORK,Ba-131,0.08)
train_Co-60_0.08:
	$(call RUN_TRAIN_WORK,Co-60,0.08)
train_Cs-137_0.08:
	$(call RUN_TRAIN_WORK,Cs-137,0.08)
train_I-125_0.08:
	$(call RUN_TRAIN_WORK,I-125,0.08)

test_Ac-225_0.08:
	$(call RUN_TEST_WORK2,Ac-225,0.08)
test_Am-241_0.08:
	$(call RUN_TEST_WORK2,Am-241,0.08)
test_Ba-131_0.08:
	$(call RUN_TEST_WORK2,Ba-131,0.08)
test_Co-60_0.08:
	$(call RUN_TEST_WORK2,Co-60,0.08)
test_Cs-137_0.08:
	$(call RUN_TEST_WORK2,Cs-137,0.08)
test_I-125_0.08:
	$(call RUN_TEST_WORK2,I-125,0.08)

run: train_Ac-225_0.08 test_Ac-225_0.08 train_Am-241_0.08 test_Am-241_0.08 train_Ba-131_0.08 test_Ba-131_0.08 train_Co-60_0.08 test_Co-60_0.08 train_Cs-137_0.08 test_Cs-137_0.08 train_I-125_0.08 test_I-125_0.08