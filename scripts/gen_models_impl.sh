# use this script to call the gen_model_impl.py script and assist in training/implementation

python gen_models_impl.py --model='VAE' --epochs=50 --debug=True

# python gen_models_impl.py --model_type='GAN' --epochs=2 --debug=True

# python gen_models_impl.py --model_type='Diffusion' --epochs=2 --debug=True

python gen_models_inference.py --onnx_name=VAEDecoder_save.onnx #--onnx_loction=

# python gen_models_inference.py --onnx_name=GenGANModel.onnx

# python gen_models_inference.py --onnx_name=GenDiffusionModel.onnx