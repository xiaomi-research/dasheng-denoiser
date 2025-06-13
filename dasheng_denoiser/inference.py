import os
import glob
import torch
import argparse
import torchaudio
from tqdm import tqdm

from dasheng_denoiser.pretrained import denoiser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denoise_inference(input_path, output_path, model_path=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model = denoiser.from_ckpt(model_path)
    model = model.to(device)

    file_list = glob.glob(os.path.join(input_path, "*"))
    fs = 16000

    for file_path in tqdm(file_list):
        y, sr = torchaudio.load(file_path)
        assert sr == fs, "sample rate must be 16000"
        assert y.size(0) == 1, "only support mono audio"
        y = y.to(device)
        y_hat = model(y)
        y_hat = y_hat.detach().cpu()
        torchaudio.save(output_path + "/" + os.path.basename(file_path), y_hat, sr)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inp_path", type=str, help="input_path", required=True, default="valid_dataset")
    parser.add_argument("-o", "--out_path", type=str, help="output_path", required=True, default="denoised_output")
    parser.add_argument("-m", "--model", type=str, help="model_path", required=False, default="None")

    args = parser.parse_args()

    input_path = args.inp_path
    output_path = args.out_path
    model_path = args.model
    denoise_inference(input_path, output_path, model_path)
